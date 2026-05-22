"""Semantic probe — tests where vector retrieval actually does its job.

For each SEMANTIC query (no entity names; describes a concept, event, or
synonym cluster), we measure:

  - Cross-backend Jaccard@20 on vector-only retrieval
    → if all 3 backends agree, vector is doing consistent semantic work
    → if they diverge, that's a real backend differential

  - Vector top-20 vs BM25 top-20 Jaccard per backend
    → low overlap = vector finds genuinely different (semantic) chunks
    → high overlap = vector is redundant with lexical search

  - Unique-to-vector chunks (in vector top-20 but not BM25 top-20)
    → quantifies the semantic-only "lift" each backend provides

Queries drawn from real pi-session search_evidence calls + Phase B+
complex scenarios, filtered to remove pure entity-name lookups.
"""
from __future__ import annotations
import json
import sys
from itertools import combinations
from pathlib import Path

import duckdb
import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.io_lance import bench_lancedb_path
from bench.models import load_embedder, get_candidate
from bench.snapshot import SNAPSHOT_SEARCH_DUCKDB


OUT = Path("/mnt/i/dev/Legal/case_kb/bench_embeddings/reports/raw/semantic_probe")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Semantic queries — drawn from real pi-session + Phase B+ scenarios.
# Entity names removed where the semantic content stands alone; kept where
# the entity is incidental to the semantic intent (e.g. "Tracy access").
# ---------------------------------------------------------------------------
QUERIES: list[dict] = [
    # --- From pi-session (semantic queries the real agent made) ---
    {"id": "valley_point_event", "category": "event_description",
     "query": "Valley Point Shopping Centre encounter Tristan November 2025",
     "intent": "Find narrative of a specific event by descriptive details"},
    {"id": "valley_point_handover", "category": "event_description",
     "query": "Valley Point pick up drop off children",
     "intent": "Find chunks about handover events near a location"},
    {"id": "parental_alienation", "category": "concept",
     "query": "parental alienation children refusing access negative perception father",
     "intent": "Find chunks discussing the legal/psychological concept"},
    {"id": "negative_characterisation", "category": "concept",
     "query": "negative characterisation father children alienation influence",
     "intent": "Find chunks describing a behavioural pattern, not specific incidents"},
    {"id": "access_denial_synonyms", "category": "synonym_paraphrase",
     "query": "access denied refused blocked cancelled children",
     "intent": "Find chunks about access denial regardless of word choice"},
    {"id": "children_refuse_visit", "category": "synonym_paraphrase",
     "query": "children don't want to see father refuse visit access",
     "intent": "Find chunks where children expressed reluctance, however phrased"},
    {"id": "tracy_limited_access_proposals", "category": "topical",
     "query": "limited meal only access proposals suspended children",
     "intent": "Find proposals that constrained access despite different specific wordings"},
    {"id": "hk_unilateral_travel", "category": "event_description",
     "query": "Hong Kong children taken without consent passports missing",
     "intent": "Find narratives of the December 2024 incident"},

    # --- From Phase B+ complex scenarios (semantic, no entity dominance) ---
    {"id": "matrimonial_home_valuation", "category": "topical",
     "query": "matrimonial home valuation arms-length sale market value division",
     "intent": "Find chunks about valuation methodology, not just the property name"},
    {"id": "father_school_visit_framing", "category": "concept",
     "query": "father visit school kindergarten approaching children negative framing harassment",
     "intent": "Find chunks discussing the legal characterisation of school visits"},
    {"id": "indirect_contribution_caregiver", "category": "concept",
     "query": "indirect contribution caregiver homemaker non-financial role sacrifice career",
     "intent": "Find chunks about the legal concept of indirect contribution"},
    {"id": "counselling_engagement_pattern", "category": "topical",
     "query": "counselling sessions in-person tele-conference engagement willingness refusal",
     "intent": "Find chunks about the pattern of engagement with counselling"},
    {"id": "gatekeeping_pattern", "category": "concept",
     "query": "gatekeeping behaviour mother exclude father co-parenting interference",
     "intent": "Find chunks about gatekeeping legal concept and its manifestations"},
    {"id": "fdr_mediation_directions", "category": "topical",
     "query": "FDR mediation directions counsellor third party suspension access",
     "intent": "Find chunks describing the mediation process and outputs"},
    {"id": "disclosure_non_compliance", "category": "topical",
     "query": "disclosure non-compliance failure produce documents adverse inference",
     "intent": "Find chunks about the disclosure dispute"},
]


BACKENDS = ["gemini-embedding-001", "gemini-embedding-2", "qwen3-embedding-8b-fp8-vllm"]


def vector_topk(backend_id: str, query: str, k: int = 20) -> list[str]:
    spec = get_candidate(backend_id)
    embedder = load_embedder(spec.id)
    qvec = embedder.encode([query])[0]
    path = bench_lancedb_path(spec.id)
    db = lancedb.connect(str(path.parent))
    table = db.open_table(spec.id)
    df = table.search(qvec).limit(k).to_pandas()
    return df["chunk_id"].tolist()


def bm25_topk(con, query: str, k: int = 20) -> list[str]:
    sql = """
    SELECT chunk_id
    FROM chunks
    WHERE fts_main_chunks.match_bm25(chunk_id, ?) IS NOT NULL
    ORDER BY fts_main_chunks.match_bm25(chunk_id, ?) DESC
    LIMIT ?
    """
    rows = con.execute(sql, [query, query, k]).fetchall()
    return [r[0] for r in rows]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    K = 20
    con = duckdb.connect(str(SNAPSHOT_SEARCH_DUCKDB), read_only=True)

    results: dict = {}
    for q in QUERIES:
        qid = q["id"]
        qstr = q["query"]
        per_backend: dict = {}
        for backend in BACKENDS:
            top = vector_topk(backend, qstr, k=K)
            per_backend[backend] = top
        bm25 = bm25_topk(con, qstr, k=K)

        bs = {b: set(t) for b, t in per_backend.items()}
        bm25_set = set(bm25)

        # Pairwise vector Jaccard
        v_jacc = {}
        for a, b in combinations(BACKENDS, 2):
            v_jacc[f"{a.split('-')[-1]}↔{b.split('-')[-1]}"] = jaccard(bs[a], bs[b])

        # Vector-vs-BM25 Jaccard per backend
        vb_jacc = {b: jaccard(bs[b], bm25_set) for b in BACKENDS}
        # Unique-to-vector chunks (in vector top-20 but not BM25 top-20)
        unique_to_vec = {b: sorted(bs[b] - bm25_set) for b in BACKENDS}

        results[qid] = {
            "query": qstr,
            "category": q["category"],
            "intent": q["intent"],
            "top20_vector": per_backend,
            "top20_bm25": bm25,
            "pairwise_vector_jaccard": v_jacc,
            "vector_vs_bm25_jaccard": vb_jacc,
            "unique_to_vector_counts": {b: len(unique_to_vec[b]) for b in BACKENDS},
        }
        print(f"\n[{qid:35s}] ({q['category']})")
        print(f"  Vector Jaccard:   " + " | ".join(f"{k}={v:.2f}" for k, v in v_jacc.items()))
        print(f"  Vec-vs-BM25:      V1={vb_jacc[BACKENDS[0]]:.2f}  V2={vb_jacc[BACKENDS[1]]:.2f}  Qwen={vb_jacc[BACKENDS[2]]:.2f}")
        print(f"  Vec-only counts:  V1={len(unique_to_vec[BACKENDS[0]])}/{K}  V2={len(unique_to_vec[BACKENDS[1]])}/{K}  Qwen={len(unique_to_vec[BACKENDS[2]])}/{K}")

    out_path = OUT / "semantic_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # ---------------- Aggregate ----------------
    print("\n=== Aggregate ===")
    import numpy as np
    # mean pairwise vector Jaccard per category
    by_cat: dict[str, list[float]] = {}
    for r in results.values():
        cat = r["category"]
        by_cat.setdefault(cat, [])
        by_cat[cat].extend(r["pairwise_vector_jaccard"].values())
    print("Mean pairwise vector Jaccard@20 by category (higher = backends agree):")
    for cat, vals in by_cat.items():
        print(f"  {cat:22s} mean={np.mean(vals):.3f}  n_pairs={len(vals)}")

    # mean Vec-vs-BM25 Jaccard per backend across all queries
    print("\nMean Vec-vs-BM25 Jaccard@20 per backend (lower = vector finds different stuff):")
    for b in BACKENDS:
        vals = [r["vector_vs_bm25_jaccard"][b] for r in results.values()]
        print(f"  {b:35s} mean={np.mean(vals):.3f}")

    # mean unique-to-vector count per backend
    print("\nMean unique-to-vector count per backend (out of 20):")
    for b in BACKENDS:
        vals = [r["unique_to_vector_counts"][b] for r in results.values()]
        print(f"  {b:35s} mean={np.mean(vals):.2f}")

    summary = {
        "k": K,
        "n_queries": len(QUERIES),
        "by_category_pairwise_vector_jaccard": {cat: float(np.mean(vals)) for cat, vals in by_cat.items()},
        "mean_vec_vs_bm25_jaccard": {b: float(np.mean([r["vector_vs_bm25_jaccard"][b] for r in results.values()])) for b in BACKENDS},
        "mean_unique_to_vector_count": {b: float(np.mean([r["unique_to_vector_counts"][b] for r in results.values()])) for b in BACKENDS},
    }
    (OUT / "semantic_probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT / 'semantic_probe_summary.json'}")


if __name__ == "__main__":
    main()
