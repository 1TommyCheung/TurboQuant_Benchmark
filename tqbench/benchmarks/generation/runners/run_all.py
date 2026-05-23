"""Orchestrator: quality → speed → ttft → report."""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from pathlib import Path

from tqbench.benchmarks.generation.models import load_registry, quality_groups

BENCH_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _run(*cmd: str) -> int:
    log.info(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=BENCH_ROOT).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="Subset of model_ids")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-ttft", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

    models = args.models or [c.id for c in load_registry()]
    log.info(f"Pipeline begins. Models: {models}")

    if not args.skip_quality:
        for gname in quality_groups():
            _run(sys.executable, "-m",
                 "tqbench.benchmarks.generation.runners.eval_quality",
                 "--group", gname)

    if not args.skip_speed:
        for m in models:
            _run(sys.executable, "-m",
                 "tqbench.benchmarks.generation.runners.eval_speed",
                 "--model", m)

    if not args.skip_ttft:
        for m in models:
            _run(sys.executable, "-m",
                 "tqbench.benchmarks.generation.runners.eval_ttft_longctx",
                 "--model", m)

    if not args.skip_report:
        _run(sys.executable, "-m",
             "tqbench.benchmarks.generation.report")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
