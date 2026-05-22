"""CLI search shim for Phase B multi-turn agent replay.

Subagents call this via Bash to retrieve evidence from a specific backend's
bench LanceDB (frozen snapshot). Returns a compact JSON of top-k chunks.

Usage:
  python -m bench_phase_b.bench_search \
      --backend gemini-embedding-001 \
      --query "child access yuqi 31 March" \
      --k 20 \
      [--mode hybrid|vector|bm25]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import duckdb
import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.io_lance import bench_lancedb_path, read_prod_chunks
from bench.models import load_embedder, get_candidate
from bench.snapshot import SNAPSHOT_SEARCH_DUCKDB, assert_snapshot_present
from bench.stack import vector_only_retrieve, bm25_retrieve, rrf_fuse


_CHUNK_LOOKUP_CACHE = None


def _chunk_lookup() -> dict[str, dict]:
    """Load and cache chunk metadata + text from the snapshot.

    Returns dict mapping chunk_id → {source_type, party_from, date_sgt,
    legal_issues, is_privileged, in_scope, snippet}.
    """
    global _CHUNK_LOOKUP_CACHE
    if _CHUNK_LOOKUP_CACHE is None:
        df = read_prod_chunks()
        df = df.set_index("chunk_id")
        _CHUNK_LOOKUP_CACHE = df.to_dict("index")
    return _CHUNK_LOOKUP_CACHE


def _format_hit(chunk_id: str, lookup: dict) -> dict:
    row = lookup.get(chunk_id) or {}
    text = row.get("chunk_text") or ""
    return {
        "chunk_id": chunk_id,
        "source_type": row.get("source_type"),
        "party_from": row.get("party_from"),
        "date_sgt": str(row.get("date_sgt") or ""),
        "is_privileged": bool(row.get("is_privileged", False)),
        "in_scope": row.get("in_scope"),
        "snippet": text[:600],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    help="Model ID from config/models.yaml (e.g. gemini-embedding-001)")
    ap.add_argument("--query", required=True, help="Free-text query")
    ap.add_argument("--k", type=int, default=20, help="Top-k to return")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    ap.add_argument("--candidates", type=int, default=50,
                    help="Candidate pool size per leg before RRF fusion")
    args = ap.parse_args()

    assert_snapshot_present()
    spec = get_candidate(args.backend)

    # Open bench LanceDB for this backend
    bench_path = bench_lancedb_path(spec.id)
    db = lancedb.connect(str(bench_path.parent))
    table = db.open_table(spec.id)

    # Embed query (mode-dependent)
    vec_ids: list[str] = []
    if args.mode in ("hybrid", "vector"):
        embedder = load_embedder(spec.id)
        qvec = embedder.encode([args.query])[0]
        vec_ids = vector_only_retrieve(table, qvec, k=args.candidates)

    # BM25 leg (snapshot DuckDB)
    bm25_ids: list[str] = []
    if args.mode in ("hybrid", "bm25"):
        con = duckdb.connect(str(SNAPSHOT_SEARCH_DUCKDB), read_only=True)
        try:
            bm25_ids = bm25_retrieve(con, args.query, k=args.candidates)
        finally:
            con.close()

    # Fuse + hydrate
    if args.mode == "hybrid":
        final_ids = rrf_fuse([vec_ids, bm25_ids], k=args.k) if bm25_ids else vec_ids[: args.k]
    elif args.mode == "vector":
        final_ids = vec_ids[: args.k]
    else:
        final_ids = bm25_ids[: args.k]

    lookup = _chunk_lookup()
    hits = [_format_hit(cid, lookup) for cid in final_ids]

    out = {
        "backend": spec.id,
        "mode": args.mode,
        "query": args.query,
        "k": args.k,
        "n_hits": len(hits),
        "hits": hits,
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
