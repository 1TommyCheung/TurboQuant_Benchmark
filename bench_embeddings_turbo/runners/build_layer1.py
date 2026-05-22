"""Build Layer 1 pool-and-judge qrels.

Reads:  data/eval_queries/_layer1_raw.jsonl
Reads:  indexes/{each_model}.lance  (assumes Phase 1 embedding done)
Writes: data/eval_queries/layer1_pool_judged.jsonl

Each output record:
{
  "id": "<query_id>",
  "query": "<query string>",
  "filters": {...},
  "qrels": {"<chunk_id>": grade, ...},  // graded 0-3 by Claude Sonnet
  "pool_size": 87,
}
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import lancedb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.io_lance import read_prod_chunks, bench_lancedb_path
from bench.models import load_registry, load_embedder, get_candidate
from bench.pool_judge import judge_pair

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries"
INDEXES = Path(__file__).resolve().parents[1] / "indexes"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=DATA / "_layer1_raw.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "layer1_pool_judged.jsonl")
    ap.add_argument("--candidates-per-query", type=int, default=20,
                    help="Top-k from each model to pool")
    args = ap.parse_args()

    log.info("Loading chunk text lookup from production...")
    prod = read_prod_chunks()
    chunk_lookup = prod.set_index("chunk_id").to_dict("index")

    raw_queries = [json.loads(l) for l in args.inp.read_text().splitlines() if l.strip()]
    log.info(f"  {len(raw_queries)} queries to pool-judge")

    # Load each model's bench LanceDB (must have been embedded already)
    indexes: dict[str, "lancedb.LanceTable"] = {}
    for spec in load_registry():
        path = bench_lancedb_path(spec.id)
        if not path.exists():
            log.warning(f"Skipping {spec.id}: no bench index at {path}")
            continue
        db = lancedb.connect(str(path.parent))
        indexes[spec.id] = db.open_table(spec.id)

    out: list[dict] = []
    for q_i, q in enumerate(raw_queries):
        log.info(f"[{q_i+1}/{len(raw_queries)}] {q['query'][:80]}")

        # Pool top-K from each model
        pool: set[str] = set()
        for model_id, tbl in indexes.items():
            embedder = load_embedder(model_id)
            qvec = embedder.encode([q["query"]])[0]
            df = tbl.search(qvec).limit(args.candidates_per_query).to_pandas()
            pool.update(df["chunk_id"].tolist())

        # Judge each
        qrels: dict[str, int] = {}
        for chunk_id in pool:
            chunk = chunk_lookup.get(chunk_id)
            if not chunk:
                continue
            grade = judge_pair(q["query"], chunk, filters=q.get("filters"))
            qrels[chunk_id] = grade

        out.append({
            "id": q["id"],
            "query": q["query"],
            "filters": q.get("filters", {}),
            "qrels": qrels,
            "pool_size": len(pool),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    log.info(f"Wrote {len(out)} judged queries to {args.out}")


if __name__ == "__main__":
    main()
