"""Long-context TTFT isolation test.

Measures prefill time at 1K, 4K, 8K, 32K, 128K input tokens.
Sends max_tokens=1 to isolate prefill from decode.
Concurrency=1 to avoid batching effects.

This is where PFlash's sublinear prefill should be visible.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import httpx

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import get_candidate, load_registry
from tqbench.benchmarks.generation.speed_metrics import aggregate_ttft_results
from tqbench.benchmarks.generation.vram import VRAMSampler

BENCH_ROOT = Path(__file__).resolve().parents[1]
DATA = BENCH_ROOT / "data"
REPORTS = BENCH_ROOT / "reports" / "raw"
CONTEXT_LENGTHS = [1024, 4096, 8192, 32768, 131072]
REPS_PER_LENGTH = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_longctx_prompts() -> dict[int, list[dict]]:
    """Load prompts grouped by target_input_tokens."""
    p = DATA / "prompts_longctx.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Generate long-context prompts first.")
    by_length: dict[int, list[dict]] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        target = entry.get("target_input_tokens", 0)
        by_length.setdefault(target, []).append(entry)
    return by_length


def _measure_ttft(client: httpx.Client, model: str, prompt: dict) -> float:
    """Send a streaming request with max_tokens=1 and return TTFT in seconds."""
    messages = prompt["messages"]
    t0 = time.perf_counter()
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": model, "messages": messages,
              "max_tokens": 1, "temperature": 0.0, "stream": True},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content") is not None:
                return time.perf_counter() - t0
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=CONTEXT_LENGTHS)
    args = ap.parse_args()

    spec = get_candidate(args.model)
    server = get_server(spec.server)
    host = server["host"]

    prompts_by_length = _load_longctx_prompts()
    client = httpx.Client(base_url=host, timeout=600)

    r = client.get("/v1/models")
    if r.status_code != 200:
        raise RuntimeError(f"Server not reachable at {host}")

    vram = VRAMSampler()
    vram.start()

    results_by_length: dict[str, dict] = {}
    for ctx_len in args.lengths:
        available = prompts_by_length.get(ctx_len, [])
        if not available:
            log.warning(f"No prompts for {ctx_len} tokens, skipping")
            continue
        reps = min(REPS_PER_LENGTH, len(available))
        log.info(f"Context {ctx_len:,} tokens — {reps} reps...")

        ttfts: list[float] = []
        for i in range(reps):
            prompt = available[i % len(available)]
            ttft = _measure_ttft(client, spec.model_name, prompt)
            ttfts.append(ttft)
            log.info(f"  rep {i+1}/{reps}: TTFT={ttft*1000:.0f}ms")

        agg = aggregate_ttft_results(ttfts)
        results_by_length[str(ctx_len)] = agg
        log.info(f"  median={agg['median_s']*1000:.0f}ms  p95={agg['p95_s']*1000:.0f}ms")

    peak_vram_mb = vram.stop()
    client.close()

    date = dt.date.today().isoformat()
    out = {
        "model_id": spec.id,
        "server": spec.server,
        "spec_decode": spec.spec_decode,
        "spec_prefill": spec.spec_prefill,
        "peak_vram_mb": peak_vram_mb,
        "by_context_length": results_by_length,
    }
    out_path = REPORTS / f"{date}_{spec.id}_ttft_longctx.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
