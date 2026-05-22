"""Build Layer 2: perturbed (2a) + filtered synthetic (2b) queries.

Layer 2a: perturb Layer 1 + Layer 3 → ~3000 queries.
Layer 2b: Gemini + Claude generate from sampled chunks → leakage-filter → ~1000 queries.

Outputs:
  data/eval_queries/layer2a_perturbed.jsonl
  data/eval_queries/layer2b_synthetic.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.perturbations import perturb_all

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_samples" / "stratified_50k.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_2a(seed: int = 42) -> list[dict]:
    """Perturb Layer 1 + Layer 3 queries to ~3K Layer-2a queries."""
    base_records: list[dict] = []
    for fname in ("_layer1_raw.jsonl", "layer3_handcrafted.jsonl"):
        p = DATA / fname
        if not p.exists():
            log.warning(f"Missing {p}, skipping")
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                base_records.append(json.loads(line))

    out: list[dict] = []
    for i, rec in enumerate(base_records):
        variants = perturb_all(rec["query"], seed=seed + i)
        for j, v in enumerate(variants):
            out.append({
                "id": f"L2a-{i:04d}-{j:02d}",
                "query": v,
                "base_id": rec["id"],
                "base_query": rec["query"],
                "perturbation_index": j,
            })
    log.info(f"Generated {len(out)} Layer-2a perturbed queries from {len(base_records)} base queries")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-2b", action="store_true",
                    help="Skip Layer 2b LLM generation (just regenerate 2a)")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    rows = build_2a(seed=args.seed)
    out_path = DATA / "layer2a_perturbed.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log.info(f"Wrote {out_path}")

    if not args.skip_2b:
        log.info("Layer 2b is built by `build_layer2b.py` separately (requires LLM API calls).")


if __name__ == "__main__":
    main()
