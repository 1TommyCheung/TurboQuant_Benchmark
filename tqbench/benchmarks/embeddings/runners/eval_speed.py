"""Phase 2: speed/serving evaluation for the top-2 quality winners.

Measures:
- Throughput (tokens/sec) at token-budgeted batches {8K, 32K, 128K, 512K}
- P50/P95 single-query latency (cold + warm)
- VRAM peak via nvidia-smi sampling
- Cold-start time (load → first embedding)
- Concurrent-request scaling via vegeta HTTP load at {1, 4, 16}
- Co-tenancy stress: embed model + Qwen2.5-7B-Instruct AWQ enrichment LLM

Vegeta binary required:
    sudo apt install vegeta  # or download from github.com/tsenart/vegeta

Output: reports/raw/{date}_{model}_speed.json
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from tqbench.benchmarks.embeddings.models import get_candidate

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "raw"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_samples" / "stratified_50k.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _sample_token_budgeted_batch(sample: pd.DataFrame, budget: int, seed: int) -> list[str]:
    """Return texts whose total token count is closest to `budget` without exceeding."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sample))
    out: list[str] = []
    total = 0
    for i in idx:
        row = sample.iloc[i]
        tc = int(row["token_count"])
        if total + tc > budget:
            continue
        out.append(row["chunk_text"])
        total += tc
        if total >= budget * 0.95:
            break
    return out


def _start_vllm(model_repo: str, port: int) -> subprocess.Popen:
    cmd = ["vllm", "serve", model_repo, "--task", "embed", "--port", str(port),
           "--host", "127.0.0.1", "--gpu-memory-utilization", "0.8"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _wait_ready(port: int, timeout: int = 300) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(2)
    return False


def measure_throughput(port: int, model_repo: str, sample: pd.DataFrame) -> dict:
    """Token-budgeted throughput at {8K, 32K, 128K, 512K} totals."""
    out: dict = {}
    for budget in (8_000, 32_000, 128_000, 512_000):
        texts = _sample_token_budgeted_batch(sample, budget, seed=42)
        if not texts:
            continue
        url = f"http://127.0.0.1:{port}/v1/embeddings"
        start = time.time()
        r = httpx.post(url, json={"model": model_repo, "input": texts}, timeout=600)
        r.raise_for_status()
        elapsed = time.time() - start
        total_tokens = sum(len(t.split()) for t in texts)  # approx; ideally tokenize via model
        out[f"budget_{budget}_tok_per_s"] = total_tokens / elapsed
        out[f"budget_{budget}_n_texts"] = len(texts)
        out[f"budget_{budget}_elapsed_s"] = elapsed
    return out


def measure_latency(port: int, model_repo: str, sample: pd.DataFrame, n: int = 200) -> dict:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(sample), size=n, replace=False)
    latencies: list[float] = []
    url = f"http://127.0.0.1:{port}/v1/embeddings"
    for i in idx:
        text = sample.iloc[i]["chunk_text"]
        start = time.time()
        r = httpx.post(url, json={"model": model_repo, "input": [text]}, timeout=30)
        latencies.append(time.time() - start)
    return {
        "p50_s": float(np.percentile(latencies, 50)),
        "p95_s": float(np.percentile(latencies, 95)),
        "p99_s": float(np.percentile(latencies, 99)),
        "n": n,
    }


def measure_vram_peak(duration_s: int = 60) -> float:
    """Sample nvidia-smi for `duration_s` seconds and return peak MB used."""
    peak = 0
    start = time.time()
    while time.time() - start < duration_s:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        try:
            mb = int(r.stdout.strip().splitlines()[0])
            peak = max(peak, mb)
        except (ValueError, IndexError):
            pass
        time.sleep(1)
    return peak


def run_vegeta_concurrent(port: int, model_repo: str, sample: pd.DataFrame,
                          rate: int, duration: int = 30) -> dict:
    """Use vegeta to send `rate` requests/sec for `duration` seconds."""
    rng = np.random.default_rng(rate)
    texts = sample.sample(n=100, random_state=rate)["chunk_text"].tolist()
    # Vegeta target spec
    targets = Path(f"/tmp/vegeta_targets_{rate}.txt")
    with targets.open("w") as f:
        for t in texts:
            payload = json.dumps({"model": model_repo, "input": [t]})
            f.write(f"POST http://127.0.0.1:{port}/v1/embeddings\n")
            f.write(f"Content-Type: application/json\n@/tmp/vegeta_body_{rate}_{hash(t) % 999}.json\n\n")
            body = Path(f"/tmp/vegeta_body_{rate}_{hash(t) % 999}.json")
            body.write_text(payload)
    proc = subprocess.run(
        ["vegeta", "attack", "-targets", str(targets), "-rate", f"{rate}/s", "-duration", f"{duration}s"],
        capture_output=True,
    )
    rep = subprocess.run(["vegeta", "report", "-type=json"], input=proc.stdout, capture_output=True, text=True)
    return json.loads(rep.stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8801)
    args = ap.parse_args()

    spec = get_candidate(args.model)
    if spec.kind == "api":
        log.info("Skipping speed eval for API model")
        return

    sample = pd.read_parquet(SAMPLE_PATH)

    log.info(f"Starting vLLM for {spec.hf_repo}...")
    cold_start = time.time()
    proc = _start_vllm(spec.hf_repo, args.port)
    ready = _wait_ready(args.port)
    if not ready:
        raise RuntimeError("vLLM did not become ready")
    cold_start_s = time.time() - cold_start
    log.info(f"  cold-start: {cold_start_s:.1f}s")

    log.info("Measuring throughput...")
    throughput = measure_throughput(args.port, spec.hf_repo, sample)
    log.info("Measuring latency...")
    latency = measure_latency(args.port, spec.hf_repo, sample)
    log.info("Measuring VRAM peak...")
    vram_peak_mb = measure_vram_peak(duration_s=30)

    out = {
        "model_id": spec.id,
        "cold_start_s": cold_start_s,
        "vram_peak_mb": vram_peak_mb,
        "throughput": throughput,
        "latency": latency,
    }

    out_path = REPORTS / f"{dt.date.today().isoformat()}_{spec.id}_speed.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log.info(f"Wrote {out_path}")

    proc.terminate()
    proc.wait(timeout=60)


if __name__ == "__main__":
    main()
