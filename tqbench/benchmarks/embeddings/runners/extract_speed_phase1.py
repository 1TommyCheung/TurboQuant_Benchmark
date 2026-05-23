"""Parse existing embed_corpus + eval_quality logs to extract Phase 1 speed metrics.

Phase 2 (vLLM/Ollama with vegeta) is a separate runner. Phase 1 speeds come
from the harness's own logs:

  * Embedding throughput  = final 'rate=X chunks/s' from the embed_corpus log
  * Approx embed wall     = rows / final_rate (logs are tail-truncated, so we
                            derive wall time from the steady-state rate)
  * Query encode wall     = bracketed by first eval progress line and the
                            'Wrote ..._quality.json' line, then scaled to the
                            full query count
  * Approx per-query ms   = encode wall * 1000 / n_queries

Identifies which model a log belongs to by the unambiguous output paths:
  'Done. Wrote NNNN rows to .../indexes/MODEL.lance'   -> embed event
  'Wrote .../reports/raw/DATE_MODEL_quality.json'      -> eval event

Writes: reports/raw/{date}_{model}_speed_phase1.json per model.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import re
from pathlib import Path

from tqbench.benchmarks.embeddings.models import load_registry

BENCH_ROOT = Path(__file__).resolve().parents[1]
RAW = BENCH_ROOT / "reports" / "raw"
DEFAULT_TASKS_DIR = Path("/tmp/claude-1000/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/tasks")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"
RATE_RE = re.compile(rf"{TS} \[INFO\]   [\d,]+/[\d,]+\s+rate=([\d.]+) chunks/s")
DONE_EMBED_RE = re.compile(rf"{TS} \[INFO\] Done\. Wrote (\d+) rows to .*?/indexes/([a-zA-Z0-9._-]+)\.lance")
DONE_EVAL_RE = re.compile(rf"{TS} \[INFO\] Wrote .+/reports/raw/[\d-]+_([a-zA-Z0-9._-]+)_quality\.json")
EVAL_PROGRESS_RE = re.compile(rf"{TS} \[INFO\]   (\d+)/(\d+)\b")


def _parse_ts(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _scan(tasks_dir: Path) -> dict:
    """Walk all .output files. For each model_id, record best (embed, eval) data."""
    if not tasks_dir.exists():
        log.warning(f"Tasks dir not found: {tasks_dir}")
        return {}

    out: dict[str, dict] = {}
    files = sorted(tasks_dir.glob("*.output"))
    log.info(f"  scanning {len(files)} log files")

    for log_file in files:
        text = log_file.read_text(errors="replace")

        # ---- EMBED event ----
        embed_done = DONE_EMBED_RE.search(text)
        if embed_done:
            model_id = embed_done.group(3)
            rows = int(embed_done.group(2))
            rate_matches = list(RATE_RE.finditer(text))
            if rate_matches:
                final_rate = float(rate_matches[-1].group(2))
                # Steady-state wall = rows / final_rate
                approx_wall_s = rows / final_rate if final_rate else 0.0
                t_first = _parse_ts(rate_matches[0].group(1))
                t_done = _parse_ts(embed_done.group(1))
                log_span_s = (t_done - t_first).total_seconds()
                slot = out.setdefault(model_id, {})
                slot["embed"] = {
                    "rows": rows,
                    "final_rate_chunks_per_s": final_rate,
                    "approx_wall_seconds": approx_wall_s,
                    "approx_wall_minutes": approx_wall_s / 60,
                    "log_span_seconds": log_span_s,
                    "log_file": str(log_file),
                }

        # ---- EVAL event ----
        eval_done = DONE_EVAL_RE.search(text)
        if eval_done:
            model_id = eval_done.group(2)
            progress = list(EVAL_PROGRESS_RE.finditer(text))
            if progress:
                t_first_prog = _parse_ts(progress[0].group(1))
                t_done = _parse_ts(eval_done.group(1))
                wall_first_to_done_s = (t_done - t_first_prog).total_seconds()
                n_total = int(progress[-1].group(3))
                # Number of queries seen between first and final progress lines
                n_first = int(progress[0].group(2))
                seen = n_total - n_first
                # Extrapolate to full loop wall = wall_first_to_done * (n_total / seen)
                full_wall_s = wall_first_to_done_s * (n_total / seen) if seen else wall_first_to_done_s
                slot = out.setdefault(model_id, {})
                slot["eval"] = {
                    "n_queries": n_total,
                    "approx_wall_seconds": full_wall_s,
                    "approx_wall_minutes": full_wall_s / 60,
                    "queries_per_s": n_total / full_wall_s if full_wall_s else 0,
                    "approx_per_query_ms": 1000 * full_wall_s / n_total if n_total else 0,
                    "log_file": str(log_file),
                }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    log.info(f"Scanning {args.tasks_dir}...")
    parsed = _scan(args.tasks_dir)
    log.info(f"  found data for {len(parsed)} model(s): {sorted(parsed)}")

    registry = {c.id: c for c in load_registry()}

    for model_id, data in parsed.items():
        spec = registry.get(model_id)
        out = {
            "model_id": model_id,
            "precision": spec.precision if spec else None,
            "dim": spec.dim if spec else None,
            "max_ctx_tokens": spec.max_ctx_tokens if spec else None,
            "embed_phase1": data.get("embed"),
            "eval_phase1": data.get("eval"),
        }
        out_path = RAW / f"{args.date}_{model_id}_speed_phase1.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        log.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
