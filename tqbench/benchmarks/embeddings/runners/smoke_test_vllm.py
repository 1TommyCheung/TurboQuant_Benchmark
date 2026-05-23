"""Phase 0: verify each non-API model loads under vLLM and produces non-NaN vectors.

Writes: reports/raw/phase0_vllm_smoke.json
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import httpx

from tqbench.benchmarks.embeddings.models import load_registry

OUT_PATH = Path(__file__).resolve().parents[1] / "reports" / "raw" / "phase0_vllm_smoke.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SAMPLE_TEXTS = [
    "did lee & lee send a letter in february 2026",
    "interim judgment ancillary matters financial disclosure",
    "whatsapp message about children school pickup arrangement",
]


def _start_vllm(model: str, port: int) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", model,
        "--task", "embed",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "8192",
    ]
    log.info(f"Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _wait_ready(port: int, timeout: int = 180) -> bool:
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


def _try_embed(port: int, model_repo: str) -> dict:
    url = f"http://127.0.0.1:{port}/v1/embeddings"
    try:
        r = httpx.post(url, json={"model": model_repo, "input": SAMPLE_TEXTS}, timeout=60)
        r.raise_for_status()
        data = r.json()
        vecs = np.array([d["embedding"] for d in data["data"]])
        return {
            "ok": True,
            "n": vecs.shape[0],
            "dim": vecs.shape[1],
            "any_nan": bool(np.isnan(vecs).any()),
            "any_zero_norm": bool((np.linalg.norm(vecs, axis=1) == 0).any()),
        }
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()

    results: list[dict] = []
    for spec in load_registry():
        if spec.kind == "api":
            results.append({"model_id": spec.id, "skipped": "api-model"})
            continue
        log.info(f"Smoke testing {spec.id} ({spec.hf_repo})...")
        proc = _start_vllm(spec.hf_repo, args.port)
        ready = _wait_ready(args.port, timeout=300)
        if not ready:
            results.append({
                "model_id": spec.id,
                "ok": False,
                "error": "vllm did not become ready within 300s",
            })
            proc.terminate()
            proc.wait(timeout=30)
            continue
        result = _try_embed(args.port, spec.hf_repo)
        result["model_id"] = spec.id
        results.append(result)
        proc.terminate()
        proc.wait(timeout=60)
        time.sleep(5)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    log.info(f"Wrote {OUT_PATH}")
    for r in results:
        log.info(f"  {r}")


if __name__ == "__main__":
    main()
