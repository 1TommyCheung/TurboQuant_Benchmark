"""End-to-end retrieval stack copy used for evaluation.

Important: this DOES NOT import or modify production code. It re-implements
the hybrid pipeline (vector + BM25 + RRF + Gemini rerank) just well enough
for evaluation, against the bench LanceDB indexes.
"""
from __future__ import annotations
from typing import Any
import numpy as np


def vector_only_retrieve(table, query_vec: np.ndarray, k: int) -> list[str]:
    """Top-k chunk_ids by vector similarity from a bench LanceDB table."""
    df = table.search(query_vec).limit(k).to_pandas()
    return df["chunk_id"].tolist()


def bm25_retrieve(duckdb_con, query: str, k: int) -> list[str]:
    """Top-k chunk_ids by BM25 from the production DuckDB FTS index.

    Reads read-only from the production FTS table.
    """
    sql = """
    SELECT chunk_id, fts_main_chunks.match_bm25(chunk_id, ?) AS score
    FROM chunks
    WHERE score IS NOT NULL
    ORDER BY score DESC
    LIMIT ?
    """
    rows = duckdb_con.execute(sql, [query, k]).fetchall()
    return [r[0] for r in rows]


def rrf_fuse(lists: list[list[str]], k: int = 10, c: int = 60) -> list[str]:
    """Reciprocal Rank Fusion of multiple ranked lists."""
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def hybrid_retrieve(
    vec_table,
    duckdb_con,
    query: str,
    query_vec: np.ndarray,
    k: int = 10,
    candidates: int = 50,
) -> list[str]:
    """Full hybrid retrieval: vector + BM25 + RRF. Does NOT include reranker.

    Returns top-k chunk_ids after RRF fusion.
    """
    vec_ids = vector_only_retrieve(vec_table, query_vec, k=candidates)
    bm25_ids = bm25_retrieve(duckdb_con, query, k=candidates) if duckdb_con else []
    if not bm25_ids:
        return vec_ids[:k]
    return rrf_fuse([vec_ids, bm25_ids], k=k)


def rerank_with_gemini(query: str, candidate_ids: list[str], k: int = 10) -> list[str]:
    """Stub — production uses Gemini rerank. For benchmarking, we want to
    measure both with-rerank and without-rerank performance so the eval
    runner calls this only when rerank is requested.
    """
    raise NotImplementedError(
        "Production-equivalent Gemini reranker call must be wired in eval_quality.py "
        "or imported from tools/search if user authorises read-only import."
    )
