"""Throughput and latency benchmark at controlled concurrency levels.

For each concurrency level (1, 4, 16, 64):
- Sends 150 prompts with stream=true via async httpx
- Measures TTFT, ITL, total latency, output tokens
- Aggregates into median/p95/p99 stats
"""
from __future__ import annotations
import argparse
import asyncio
import datetime as dt
import json
import logging
import time
from pathlib import Path

import httpx
import numpy as np

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import get_candidate, load_registry
from tqbench.benchmarks.generation.speed_metrics import aggregate_stream_results
from tqbench.benchmarks.generation.vram import VRAMSampler, ServerMetricsSampler

logging.getLogger("httpx").setLevel(logging.WARNING)

BENCH_ROOT = Path(__file__).resolve().parents[1]
DATA = BENCH_ROOT / "data"
REPORTS = BENCH_ROOT / "reports" / "raw"
CONCURRENCY_LEVELS = [1, 4, 16, 64]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_prompts() -> list[dict]:
    prompts: list[dict] = []
    for name in ("prompts_short.jsonl", "prompts_medium.jsonl", "prompts_long.jsonl"):
        p = DATA / name
        if not p.exists():
            log.warning(f"Missing {p}")
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


async def _stream_one(
    client: httpx.AsyncClient,
    model: str,
    prompt: dict,
    enable_thinking: bool = False,
) -> dict:
    messages = prompt["messages"]
    max_tokens = prompt.get("max_tokens", 256)
    t0 = time.perf_counter()
    ttft = 0.0
    itl_ms: list[float] = []
    chunks: list[str] = []
    last_time = t0
    first_seen = False
    prompt_tokens = 0
    completion_tokens = 0

    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    async with client.stream(
        "POST", "/v1/chat/completions",
        json=payload,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content is None:
                continue
            now = time.perf_counter()
            if not first_seen:
                ttft = now - t0
                first_seen = True
            else:
                itl_ms.append((now - last_time) * 1000)
            last_time = now
            chunks.append(content)

    total = time.perf_counter() - t0
    return {
        "id": prompt["id"],
        "ttft_s": ttft,
        "itl_ms": itl_ms,
        "completion_tokens": completion_tokens or len(chunks),
        "prompt_tokens": prompt_tokens,
        "total_time_s": total,
        "text": "".join(chunks),
    }


async def run_concurrency_level(
    host: str, model: str, prompts: list[dict], concurrency: int,
) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=host, timeout=300, limits=limits) as client:

        async def _bounded(prompt: dict) -> dict:
            async with sem:
                return await _stream_one(client, model, prompt)

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(_bounded(p)) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.perf_counter() - t0

    valid = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        log.warning(f"  {len(errors)} requests failed")
    return valid, wall_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=CONCURRENCY_LEVELS)
    args = ap.parse_args()

    spec = get_candidate(args.model)
    server = get_server(spec.server)
    host = server["host"]

    prompts = _load_prompts()
    if not prompts:
        log.error("No prompts found in data/. Generate them first.")
        return
    log.info(f"Loaded {len(prompts)} prompts")

    vram = VRAMSampler()
    vram.start()
    metrics = ServerMetricsSampler(host)
    metrics.start()

    all_results: dict[int, dict] = {}
    for conc in args.concurrency:
        log.info(f"Concurrency {conc}...")
        raw, wall = asyncio.run(run_concurrency_level(host, spec.model_name, prompts, conc))

        from tqbench.benchmarks.generation.clients import StreamResult
        stream_results = [
            StreamResult(
                text=r["text"], prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                ttft_s=r["ttft_s"], itl_ms=r["itl_ms"], total_time_s=r["total_time_s"],
            )
            for r in raw
        ]
        agg = aggregate_stream_results(stream_results, wall)
        agg["peak_vram_mb"] = vram.mark()
        agg["server_metrics"] = metrics.mark()
        all_results[conc] = agg
        kv = agg["server_metrics"]
        log.info(f"  throughput={agg['throughput_tok_s']:.1f} tok/s  "
                 f"ttft_p50={agg['ttft_median_s']*1000:.0f}ms  "
                 f"itl_p50={agg['itl_median_ms']:.1f}ms  "
                 f"vram_peak={agg['peak_vram_mb']}MB  "
                 f"kv_cache={kv['kv_cache_usage_peak_pct']:.1f}%")

    peak_vram_mb = vram.stop()
    metrics.stop()

    date = dt.date.today().isoformat()
    out = {
        "model_id": spec.id,
        "server": spec.server,
        "spec_decode": spec.spec_decode,
        "spec_prefill": spec.spec_prefill,
        "peak_vram_mb": peak_vram_mb,
        "by_concurrency": {str(k): v for k, v in all_results.items()},
    }
    out_path = REPORTS / f"{date}_{spec.id}_speed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
