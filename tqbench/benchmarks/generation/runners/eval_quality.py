"""Quality evaluation via lm-evaluation-harness tinyBenchmarks.

Shells out to the lm_eval CLI rather than reimplementing eval logic.
Runs once per quality_group (fp8, q8) — not per config.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import get_candidate, quality_groups

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "raw"
TASKS = "tinyMMLU,tinyHellaswag,tinyArc,tinyWinogrande"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_quality_eval(model_id: str) -> Path:
    spec = get_candidate(model_id)
    server = get_server(spec.server)
    host = server["host"]
    date = dt.date.today().isoformat()
    out_dir = REPORTS / f"{date}_{model_id}_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", f"model={spec.model_name},base_url={host}/v1,tokenizer_backend=huggingface",
        "--tasks", TASKS,
        "--batch_size", "1",
        "--output_path", str(out_dir),
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"lm_eval failed:\n{result.stderr}")
        raise RuntimeError(f"lm_eval exited with code {result.returncode}")

    log.info(f"Quality results written to {out_dir}")
    return out_dir


def parse_quality_results(result_dir: Path) -> dict:
    """Parse lm_eval output directory into a summary dict."""
    results_file = None
    for f in result_dir.rglob("results.json"):
        results_file = f
        break
    if not results_file:
        for f in result_dir.glob("*.json"):
            results_file = f
            break
    if not results_file:
        raise FileNotFoundError(f"No results.json found in {result_dir}")

    raw = json.loads(results_file.read_text())
    results = raw.get("results", raw)
    summary = {}
    for task_name, metrics in results.items():
        if isinstance(metrics, dict):
            acc = metrics.get("acc,none", metrics.get("acc_norm,none", metrics.get("acc")))
            if acc is not None:
                summary[task_name] = float(acc)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Specific model_id to eval")
    ap.add_argument("--group", help="Quality group to eval (fp8 or q8)")
    args = ap.parse_args()

    if args.model:
        out_dir = run_quality_eval(args.model)
        summary = parse_quality_results(out_dir)
        log.info(f"Scores: {json.dumps(summary, indent=2)}")
    elif args.group:
        groups = quality_groups()
        if args.group not in groups:
            log.error(f"Unknown group '{args.group}'. Known: {list(groups.keys())}")
            sys.exit(1)
        representative = groups[args.group][0]
        log.info(f"Running quality eval for group '{args.group}' using config '{representative.id}'")
        out_dir = run_quality_eval(representative.id)
        summary = parse_quality_results(out_dir)
        log.info(f"Scores: {json.dumps(summary, indent=2)}")
    else:
        for gname, specs in quality_groups().items():
            representative = specs[0]
            log.info(f"Group '{gname}': using '{representative.id}'")
            out_dir = run_quality_eval(representative.id)
            summary = parse_quality_results(out_dir)
            log.info(f"  Scores: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
