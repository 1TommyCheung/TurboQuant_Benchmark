"""Compute per-backend recall@20 against multi-judge consensus."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import lancedb
from bench.io_lance import bench_lancedb_path
from bench.models import load_embedder, get_candidate
from bench.snapshot import SNAPSHOT_SEARCH_DUCKDB


QUERIES = {
    "parental_alienation": "parental alienation children refusing access negative perception father",
    "access_denial_synonyms": "access denied refused blocked cancelled children",
    "valley_point_event": "Valley Point Shopping Centre encounter Tristan November 2025",
    "disclosure_non_compliance": "disclosure non-compliance failure produce documents adverse inference",
    "indirect_contribution_caregiver": "indirect contribution caregiver homemaker non-financial role sacrifice career",
    "valley_point_handover": "Valley Point pick up drop off children",
    "negative_characterisation": "negative characterisation father children alienation influence",
    "children_refuse_visit": "children don't want to see father refuse visit access",
    "tracy_limited_access_proposals": "limited meal only access proposals suspended children",
    "hk_unilateral_travel": "Hong Kong children taken without consent passports missing",
    "father_school_visit_framing": "father visit school kindergarten approaching children negative framing harassment",
    "counselling_engagement_pattern": "counselling sessions in-person tele-conference engagement willingness refusal",
    "gatekeeping_pattern": "gatekeeping behaviour mother exclude father co-parenting interference",
    "fdr_mediation_directions": "FDR mediation directions counsellor third party suspension access",
    "matrimonial_home_valuation": "matrimonial home valuation arms-length sale market value division",
}

BACKENDS = ["gemini-embedding-001", "gemini-embedding-2", "qwen3-embedding-8b-fp8-vllm"]

OUT = Path("/mnt/i/dev/Legal/case_kb/bench_embeddings/reports/raw/judge_scores")
SCORES_PATH = OUT / "consolidated_scores.json"


def vector_topk(backend_id: str, query: str, k: int = 20) -> list[str]:
    spec = get_candidate(backend_id)
    embedder = load_embedder(spec.id)
    qvec = embedder.encode([query])[0]
    path = bench_lancedb_path(spec.id)
    db = lancedb.connect(str(path.parent))
    table = db.open_table(spec.id)
    return table.search(qvec).limit(k).to_pandas()["chunk_id"].tolist()


def bm25_topk(con, query: str, k: int = 20) -> list[str]:
    sql = """
    SELECT chunk_id FROM chunks
    WHERE fts_main_chunks.match_bm25(chunk_id, ?) IS NOT NULL
    ORDER BY fts_main_chunks.match_bm25(chunk_id, ?) DESC
    LIMIT ?
    """
    return [r[0] for r in con.execute(sql, [query, query, k]).fetchall()]


def hybrid_topk(vec_ids: list[str], bm25_ids: list[str], k: int = 20, c: int = 60) -> list[str]:
    """RRF fusion."""
    scores: dict[str, float] = {}
    for ranked in [vec_ids, bm25_ids]:
        for rank, did in enumerate(ranked, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (c + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def main() -> None:
    scores = json.loads(SCORES_PATH.read_text())
    con = duckdb.connect(str(SNAPSHOT_SEARCH_DUCKDB), read_only=True)

    K = 20
    report: dict = {}
    for qid, query in QUERIES.items():
        per_chunk = scores[qid]
        # Three relevance thresholds
        rel_loose = {c for c, s in per_chunk.items() if s.get("mean", 0) >= 1.0}   # at least weak
        rel_consensus = {c for c, s in per_chunk.items() if s.get("mean", 0) >= 1.5}   # mean across judges suggests relevant
        rel_strict = {c for c, s in per_chunk.items() if s.get("mean", 0) >= 2.5}   # consensus highly relevant

        bm25 = bm25_topk(con, query, k=K)
        bm25_set = set(bm25)

        row = {
            "query": query,
            "pool_size": len(per_chunk),
            "n_consensus_relevant": len(rel_consensus),
            "n_strict_relevant": len(rel_strict),
            "backends": {},
            "bm25": {
                "top_k_ids": bm25,
                "recall_consensus": len(bm25_set & rel_consensus) / max(1, len(rel_consensus)),
                "recall_strict": len(bm25_set & rel_strict) / max(1, len(rel_strict)),
            },
        }

        for backend in BACKENDS:
            top = vector_topk(backend, query, k=K)
            top_set = set(top)
            hybrid = set(hybrid_topk(top, bm25, k=K))
            row["backends"][backend] = {
                "vector_top_k_ids": top,
                "vector_recall_consensus": len(top_set & rel_consensus) / max(1, len(rel_consensus)),
                "vector_recall_strict": len(top_set & rel_strict) / max(1, len(rel_strict)),
                "hybrid_recall_consensus": len(hybrid & rel_consensus) / max(1, len(rel_consensus)),
                "hybrid_recall_strict": len(hybrid & rel_strict) / max(1, len(rel_strict)),
            }

        report[qid] = row

        # Print summary table for this query
        print(f"\n=== {qid} ===")
        print(f"  Pool: {row['pool_size']} | Consensus-relevant: {row['n_consensus_relevant']} | Strictly relevant: {row['n_strict_relevant']}")
        print(f"  {'Backend':<35s} {'vec_rec':>9s} {'hyb_rec':>9s} {'vec_str':>9s} {'hyb_str':>9s}")
        print(f"  {'BM25 only':<35s} {'-':>9s} {'-':>9s} {row['bm25']['recall_consensus']:>9.3f} {row['bm25']['recall_strict']:>9.3f}")
        for b in BACKENDS:
            d = row["backends"][b]
            print(f"  {b:<35s} {d['vector_recall_consensus']:>9.3f} {d['hybrid_recall_consensus']:>9.3f} {d['vector_recall_strict']:>9.3f} {d['hybrid_recall_strict']:>9.3f}")

    out_path = OUT / "backend_recall_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
