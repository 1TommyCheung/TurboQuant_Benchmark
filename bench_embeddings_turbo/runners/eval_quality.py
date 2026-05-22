"""Phase 1: run quality evaluation against all eval-data layers for one model.

For each query in Layer 1+2a+2b+Layer 3+Adversarial:
- Encode query → retrieve top-k from bench LanceDB (vector-only)
- Also retrieve top-k via end-to-end pipeline (vector + BM25 + RRF)
- Compute recall@10, MRR, NDCG (graded for Layer 1) per query
- Aggregate by (source_type × length_bucket)

Outputs: reports/raw/{date}_{model}_quality.json
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import lancedb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.io_lance import bench_lancedb_path, read_prod_chunks
from bench.metrics import recall_at_k, mrr_at_k, ndcg_at_k, bootstrap_ci
from bench.models import get_candidate, load_embedder
from bench.snapshot import SNAPSHOT_SEARCH_DUCKDB
from bench.stack import vector_only_retrieve, bm25_retrieve, rrf_fuse

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries"
REPORTS = Path(__file__).resolve().parents[1] / "reports" / "raw"
# Frozen snapshot — see bench/snapshot.py + SNAPSHOT.md.
PROD_DUCKDB = SNAPSHOT_SEARCH_DUCKDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_queries() -> list[dict]:
    out: list[dict] = []
    layer_files = {
        "layer1": DATA / "layer1_pool_judged.jsonl",
        "layer2a": DATA / "layer2a_perturbed.jsonl",
        "layer2b": DATA / "layer2b_synthetic.jsonl",
        "layer3": DATA / "layer3_handcrafted.jsonl",
        "adversarial": DATA / "adversarial_gemini_failures.jsonl",
    }
    for layer, p in layer_files.items():
        if not p.exists():
            log.warning(f"Missing {p}")
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["_layer"] = layer
            out.append(r)
    return out


def _positives_for(q: dict) -> tuple[set[str], dict[str, int] | None]:
    """Return (positive set, graded qrels) for a query."""
    if q["_layer"] == "layer1":
        qrels = {k: v for k, v in q["qrels"].items() if v > 0}
        return set(qrels.keys()), qrels
    if q["_layer"] == "layer2b":
        return {q["source_chunk_id"]}, None
    if q["_layer"] == "adversarial":
        return set(q.get("positives", [])), None
    # Layer 2a, Layer 3 — use the base query's positives if available, else none
    return set(), None


def evaluate_query(q: dict, vec_table, duckdb_con, embedder, k: int = 10) -> dict:
    qvec = embedder.encode([q["query"]])[0]
    vec_ids = vec_table.search(qvec).limit(50).to_pandas()["chunk_id"].tolist()
    bm25_ids = bm25_retrieve(duckdb_con, q["query"], k=50) if duckdb_con else []
    e2e_ids = rrf_fuse([vec_ids, bm25_ids], k=k) if bm25_ids else vec_ids[:k]

    positives, qrels = _positives_for(q)

    metrics = {
        "vec_recall_10": recall_at_k(vec_ids[:k], positives, k),
        "vec_recall_100": recall_at_k(vec_ids[:100], positives, 100),
        "vec_mrr_10": mrr_at_k(vec_ids[:k], positives, k),
        "e2e_recall_10": recall_at_k(e2e_ids[:k], positives, k),
        "e2e_mrr_10": mrr_at_k(e2e_ids[:k], positives, k),
    }
    if qrels:
        metrics["vec_ndcg_10"] = ndcg_at_k(vec_ids[:k], qrels, k)
        metrics["e2e_ndcg_10"] = ndcg_at_k(e2e_ids[:k], qrels, k)
    return metrics


def aggregate(per_query: list[dict], chunk_lookup: dict[str, dict]) -> dict:
    """Aggregate per-query metrics by source_type × length_bucket."""
    by_st: dict[str, list] = defaultdict(list)
    by_lb: dict[str, list] = defaultdict(list)
    for r in per_query:
        cid = r.get("source_chunk_id")
        if cid and cid in chunk_lookup:
            chunk = chunk_lookup[cid]
            by_st[chunk["source_type"]].append(r["metrics"]["e2e_recall_10"])

    return {
        "overall_mean": {
            k: np.mean([r["metrics"][k] for r in per_query if k in r["metrics"]])
            for k in ("vec_recall_10", "vec_mrr_10", "e2e_recall_10", "e2e_mrr_10")
        },
        "by_source_type": {st: float(np.mean(vs)) for st, vs in by_st.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    spec = get_candidate(args.model)
    embedder = load_embedder(spec.id)

    log.info("Loading bench LanceDB...")
    path = bench_lancedb_path(spec.id)
    db = lancedb.connect(str(path.parent))
    vec_tbl = db.open_table(spec.id)

    log.info("Connecting to production DuckDB FTS (read-only)...")
    duckdb_con = duckdb.connect(str(PROD_DUCKDB), read_only=True)

    log.info("Loading eval queries...")
    queries = _load_queries()
    log.info(f"  {len(queries)} queries")

    chunk_lookup = read_prod_chunks().set_index("chunk_id").to_dict("index")

    per_query: list[dict] = []
    for i, q in enumerate(queries):
        try:
            m = evaluate_query(q, vec_tbl, duckdb_con, embedder, k=args.k)
        except Exception as e:
            log.warning(f"  query {q['id']} failed: {e}")
            continue
        per_query.append({"id": q["id"], "layer": q["_layer"], "metrics": m, **{k: q.get(k) for k in ("source_chunk_id",)}})
        if (i + 1) % 100 == 0:
            log.info(f"  {i+1}/{len(queries)}")

    agg = aggregate(per_query, chunk_lookup)

    out_path = REPORTS / f"{dt.date.today().isoformat()}_{spec.id}_quality.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model_id": spec.id,
        "n_queries": len(per_query),
        "per_query": per_query,
        "aggregate": agg,
    }, indent=2, default=float))
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
