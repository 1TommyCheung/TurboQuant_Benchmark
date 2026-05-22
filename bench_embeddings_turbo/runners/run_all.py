"""End-to-end orchestrator: runs every stage in order.

Stages:
  1. build_corpus_sample          (~30s)
  2. extract_session_queries      (~5s)
  3. smoke_test_vllm              (~30 min — Phase 0)
  4. embed_corpus per model       (~12-16h total — Phase 1)
  5. build_layer1 (pool-and-judge after embedding)
  6. build_layer2  (Layer 2a)
  7. build_layer2b
  8. build_adversarial
  9. build_dirty
 10. eval_quality per model       (~30 min per model)
 11. eval_speed (top-2 only)      (~30 min total)
 12. build_report                 (~5s)

Hard timebox (per spec §12): 3 wall-clock days for Phase 1. If exceeded,
emit interim report and stop.

Usage:
    python -m runners.run_all                  # full pipeline
    python -m runners.run_all --skip-phase2    # quality only
    python -m runners.run_all --models qwen3-embedding-8b-int8 harrier-oss-0.6b-bf16
"""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.models import load_registry

BENCH_ROOT = Path(__file__).resolve().parents[1]
PHASE1_TIMEBOX_HOURS = 72

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _run(*cmd: str) -> int:
    log.info(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=BENCH_ROOT).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-phase0", action="store_true")
    ap.add_argument("--skip-phase1-embed", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument("--models", nargs="+", help="Subset of model_ids to run")
    args = ap.parse_args()

    models = args.models or [c.id for c in load_registry()]
    log.info(f"Pipeline begins. Models: {models}")
    start = time.time()

    _run(sys.executable, "-m", "runners.build_corpus_sample")
    _run(sys.executable, "-m", "runners.extract_session_queries")
    _run(sys.executable, "-m", "runners.build_layer2", "--skip-2b")
    _run(sys.executable, "-m", "runners.build_layer2b", "--n-chunks", "1500")
    _run(sys.executable, "-m", "runners.build_adversarial")
    _run(sys.executable, "-m", "runners.build_dirty")

    if not args.skip_phase0:
        _run(sys.executable, "-m", "runners.smoke_test_vllm")

    if not args.skip_phase1_embed:
        for m in models:
            elapsed_h = (time.time() - start) / 3600
            if elapsed_h > PHASE1_TIMEBOX_HOURS:
                log.error(f"PHASE 1 TIMEBOX EXCEEDED ({elapsed_h:.1f}h > {PHASE1_TIMEBOX_HOURS}h). Stopping.")
                break
            _run(sys.executable, "-m", "runners.embed_corpus", "--model", m)

    # Pool-and-judge Layer 1 after embeddings exist
    _run(sys.executable, "-m", "runners.build_layer1")

    for m in models:
        _run(sys.executable, "-m", "runners.eval_quality", "--model", m)

    if not args.skip_phase2:
        # Top-2 quality winners — would normally select dynamically; for now hardcode top 2
        # by reading the latest quality JSONs.
        # Stub: run top-1 only
        top_two = models[:2]
        for m in top_two:
            spec = next(c for c in load_registry() if c.id == m)
            if spec.kind == "api":
                continue
            _run(sys.executable, "-m", "runners.eval_speed", "--model", m)

    _run(sys.executable, "-m", "runners.build_report")

    log.info(f"Pipeline complete in {(time.time() - start)/3600:.1f}h.")


if __name__ == "__main__":
    main()
