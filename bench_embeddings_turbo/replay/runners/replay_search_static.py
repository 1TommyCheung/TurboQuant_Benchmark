"""Phase A: replay each search_evidence call through both retrieval backends
reading exclusively from the FROZEN snapshot, compute metrics, write JSON.

Reads:
  - data/replay/session_calls.json (Task 2 output)
  - bench_embeddings/indexes/{model_id}.lance (per-backend bench LanceDB)
  - bench.snapshot.SNAPSHOT_SEARCH_DUCKDB (FTS leg)
  - bench.snapshot.SNAPSHOT_LANCEDB_PATH (chunk_text + source_type lookup)

Writes:
  - bench_embeddings/reports/raw/{date}_phase_a_static_diff.json
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import math
import sys
from pathlib import Path

import duckdb
import lancedb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bench.io_lance import BENCH_LANCEDB_ROOT
from bench.models import load_embedder, get_candidate
from bench.snapshot import (
    SNAPSHOT_LANCEDB_PATH, SNAPSHOT_SEARCH_DUCKDB, assert_snapshot_present
)
from bench.source_weights import weighted_cited_overlap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CALLS = Path(__file__).resolve().parents[1] / "data" / "replay" / "session_calls.json"
DEFAULT_OUT = BENCH_LANCEDB_ROOT.parent / "reports" / "raw" / f"{dt.date.today().isoformat()}_phase_a_static_diff.json"


def jaccard_at_k(a: list[str], b: list[str], k: int) -> float:
    A = set(a[:k])
    B = set(b[:k])
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def ndcg_overlap_at_k(a: list[str], b: list[str], k: int) -> float:
    score, ideal = 0.0, 0.0
    for i, x in enumerate(a[:k]):
        weight = 1.0 / math.log2(i + 2)
        ideal += weight
        if x in b[:k]:
            score += weight
    return score / ideal if ideal else 0.0


def divergence_score(jaccard: float, cited_weighted: float, ndcg_overlap: float) -> float:
    return 0.5 * (1 - jaccard) + 0.3 * (1 - cited_weighted) + 0.2 * (1 - ndcg_overlap)


def _vector_search(table, query_vec: np.ndarray, k: int,
                   source_type_filter: str | None = None,
                   party_filter: str | None = None,
                   date_from: str | None = None,
                   date_to: str | None = None) -> list[str]:
    df = table.search(query_vec).limit(k * 5).to_pandas()
    if source_type_filter:
        df = df[df["source_type"] == source_type_filter]
    if party_filter:
        df = df[df["party_from"] == party_filter]
    if date_from:
        df = df[df["date_sgt"] >= date_from]
    if date_to:
        df = df[df["date_sgt"] <= date_to]
    return df["chunk_id"].head(k).tolist()


def _bm25_search(duckdb_con, query: str, k: int) -> list[str]:
    try:
        rows = duckdb_con.execute(
            """
            SELECT chunk_id, fts_main_chunks.match_bm25(chunk_id, ?) AS score
            FROM chunks
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            [query, k],
        ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        log.warning(f"  BM25 fallback: {e}")
        return []


def rrf_fuse(lists: list[list[str]], k: int, c: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def _cited_with_type(cited_chunk_ids: list[str], chunk_lookup: dict) -> list[tuple[str, str]]:
    out = []
    for cid in cited_chunk_ids:
        if cid in chunk_lookup:
            out.append((cid, chunk_lookup[cid].get("source_type", "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=Path, default=DEFAULT_CALLS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--backends", nargs="+", default=["gemini-embedding-001", "harrier-oss-0.6b-bf16"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    assert_snapshot_present()

    log.info(f"Loading session calls from {args.calls}")
    payload = json.loads(args.calls.read_text())
    calls = payload["calls"]
    cited_per_turn = {int(k): v for k, v in payload["cited_per_turn"].items()}
    log.info(f"  {len(calls)} tool calls across {len(cited_per_turn)} turns")

    log.info(f"Opening snapshot DuckDB {SNAPSHOT_SEARCH_DUCKDB} read-only")
    duckdb_con = duckdb.connect(str(SNAPSHOT_SEARCH_DUCKDB), read_only=True)

    log.info(f"Building chunk lookup from snapshot LanceDB {SNAPSHOT_LANCEDB_PATH}")
    db_snap = lancedb.connect(str(SNAPSHOT_LANCEDB_PATH))
    snap_tbl = db_snap.open_table("chunks")
    chunk_lookup = (
        snap_tbl.to_pandas()[["chunk_id", "source_type"]]
        .set_index("chunk_id")
        .to_dict("index")
    )

    log.info(f"Loading {len(args.backends)} backends...")
    backend_tables = {}
    backend_embedders = {}
    for bid in args.backends:
        spec = get_candidate(bid)
        db_bench = lancedb.connect(str(BENCH_LANCEDB_ROOT))
        backend_tables[bid] = db_bench.open_table(bid)
        backend_embedders[bid] = load_embedder(bid)

    out_rows: list[dict] = []
    for i, call in enumerate(calls):
        args_dict = call["args"]
        query = args_dict.get("query") or args_dict.get("claim", "")
        turn_idx = call["turn_idx"]
        cited_this_turn = cited_per_turn.get(turn_idx, [])
        cited_wt = _cited_with_type(cited_this_turn, chunk_lookup)

        per_backend = {}
        for bid in args.backends:
            qvec = backend_embedders[bid].encode([query])[0]
            vec_ids = _vector_search(
                backend_tables[bid], qvec, args.k,
                source_type_filter=args_dict.get("source_type"),
                party_filter=args_dict.get("party"),
                date_from=args_dict.get("date_from"),
                date_to=args_dict.get("date_to"),
            )
            bm25_ids = _bm25_search(duckdb_con, query, args.k)
            e2e_ids = rrf_fuse([vec_ids, bm25_ids], k=args.k) if bm25_ids else vec_ids[:args.k]
            per_backend[bid] = {
                "vec_top_k": vec_ids[:args.k],
                "e2e_top_k": e2e_ids[:args.k],
                "cited_weighted_overlap": weighted_cited_overlap(cited_wt, set(e2e_ids)),
            }

        a, b = args.backends[0], args.backends[1]
        jacc = jaccard_at_k(per_backend[a]["e2e_top_k"], per_backend[b]["e2e_top_k"], k=args.k)
        ndcg = ndcg_overlap_at_k(per_backend[a]["e2e_top_k"], per_backend[b]["e2e_top_k"], k=args.k)
        cw_a = per_backend[a]["cited_weighted_overlap"]
        cw_b = per_backend[b]["cited_weighted_overlap"]
        div = divergence_score(jacc, cw_b, ndcg)

        out_rows.append({
            "call_idx": i,
            "turn_idx": turn_idx,
            "tool_name": call["tool_name"],
            "user_text": call.get("user_text", ""),
            "query": query,
            "filters": {k: v for k, v in args_dict.items() if k not in ("query", "claim")},
            "n_cited_in_turn": len(cited_wt),
            "backends": per_backend,
            "jaccard_at_k": jacc,
            "ndcg_overlap_at_k": ndcg,
            "cited_weighted_a": cw_a,
            "cited_weighted_b": cw_b,
            "cited_weighted_delta": cw_a - cw_b,
            "divergence_score": div,
        })

        if (i + 1) % 5 == 0:
            log.info(f"  {i+1}/{len(calls)} calls processed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "date": dt.date.today().isoformat(),
        "backends": args.backends,
        "k": args.k,
        "n_calls": len(out_rows),
        "rows": out_rows,
    }, indent=2, default=float))
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
