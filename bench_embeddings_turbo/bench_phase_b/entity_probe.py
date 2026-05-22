"""SG-specific named-entity recall probe across V1/V2/Qwen backends.

For each entity:
  1. Find the BM25-oracle set: chunks whose text contains the entity string (case-insensitive)
  2. For each backend: embed the entity as a query, retrieve top-K
  3. Compute recall@K = |top_K ∩ oracle| / |oracle|

The intent is to detect whether V1's strong entity coverage masks pipeline-design
flaws that surface on weaker-entity backends like Qwen3-FP8.

Output: per-entity, per-backend recall@K table + summary + flagged pipeline issues.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import duckdb
import lancedb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.io_lance import bench_lancedb_path, read_prod_chunks
from bench.models import load_embedder, get_candidate
from bench.snapshot import SNAPSHOT_SEARCH_DUCKDB

OUT = Path("/mnt/i/dev/Legal/case_kb/bench_embeddings/reports/raw/entity_probe")
OUT.mkdir(parents=True, exist_ok=True)


ENTITIES: list[dict] = [
    # Solicitor firms
    {"id": "lee_and_lee", "query": "Lee and Lee", "category": "firm"},
    {"id": "gjclaw", "query": "Gloria James-Civetta and Co", "category": "firm"},
    {"id": "hep", "query": "Harry Elias Partnership", "category": "firm"},
    # GJC personnel
    {"id": "sheryl_keith", "query": "Sheryl Keith", "category": "person"},
    {"id": "gloria_james", "query": "Gloria James", "category": "person"},
    {"id": "pang_chen", "query": "Pang Chen", "category": "person"},
    {"id": "yuqi_wu", "query": "Yuqi Wu", "category": "person"},
    # L&L personnel
    {"id": "amelia_ang", "query": "Amelia Ang", "category": "person"},
    # HEP personnel
    {"id": "carrie_gill", "query": "Carrie Gill", "category": "person"},
    # FJC personnel
    {"id": "dj_lim_choi_ming", "query": "DJ Lim Choi Ming", "category": "judge"},
    # Counselling agencies
    {"id": "montfortcare", "query": "MontfortCare", "category": "agency"},
    {"id": "phyllis_seah", "query": "Phyllis Seah", "category": "person"},
    {"id": "dssa", "query": "Divorce Support Specialist Agency", "category": "agency"},
    {"id": "thk_cfh", "query": "THK Centre for Family Harmony", "category": "agency"},
    # Properties
    {"id": "31_alexandra", "query": "31 Alexandra Road", "category": "address"},
    {"id": "10_shanghai", "query": "10 Shanghai Road", "category": "address"},
    {"id": "20_kay_poh", "query": "20 Kay Poh Road", "category": "address"},
    {"id": "charleston", "query": "Charleston", "category": "address"},
    {"id": "kasturina_lodge", "query": "Kasturina Lodge", "category": "address"},
    # Court docs
    {"id": "fc_sum_2273", "query": "FC/SUM 2273/2025", "category": "doc_id"},
    {"id": "form_85a", "query": "Form 85A", "category": "doc_id"},
    # Statutory
    {"id": "womens_charter", "query": "Women's Charter section 112", "category": "statute"},
    {"id": "iras_tax_ref", "query": "IRAS Tax Reference 4281657J", "category": "doc_id"},
    # Children
    {"id": "alexandra_primary", "query": "Alexandra Primary School", "category": "school"},
    {"id": "tanjong_katong", "query": "Tanjong Katong Secondary", "category": "school"},
]


BACKENDS = [
    "gemini-embedding-001",
    "gemini-embedding-2",
    "qwen3-embedding-8b-fp8-vllm",
]


def bm25_oracle(con, entity_text: str, top_k: int = 10) -> list[str]:
    """Find the TIGHT oracle: top-K BM25 hits — the chunks that lexically best
    match the entity name.

    Tighter oracle than v1 of this probe: we ask "did the vector backend find
    the most lexically-relevant chunks for this entity?" Top-10 is the standard
    benchmark for retrieval; if the backend can't get any of the top-10 BM25
    hits into its own top-20, that's a strong signal that vector and lexical
    retrieval are orthogonal for that entity.
    """
    sql = """
    SELECT chunk_id
    FROM chunks
    WHERE fts_main_chunks.match_bm25(chunk_id, ?) IS NOT NULL
    ORDER BY fts_main_chunks.match_bm25(chunk_id, ?) DESC
    LIMIT ?
    """
    rows = con.execute(sql, [entity_text, entity_text, top_k]).fetchall()
    return [r[0] for r in rows]


def backend_topk(backend_id: str, query: str, k: int = 20) -> list[str]:
    spec = get_candidate(backend_id)
    embedder = load_embedder(spec.id)
    qvec = embedder.encode([query])[0]
    path = bench_lancedb_path(spec.id)
    db = lancedb.connect(str(path.parent))
    table = db.open_table(spec.id)
    df = table.search(qvec).limit(k).to_pandas()
    return df["chunk_id"].tolist()


def main() -> None:
    K = 20
    ORACLE_K = 10  # tightened from 100
    con = duckdb.connect(str(SNAPSHOT_SEARCH_DUCKDB), read_only=True)

    results: dict = {}
    for entity in ENTITIES:
        eid = entity["id"]
        q = entity["query"]
        oracle = set(bm25_oracle(con, q, top_k=ORACLE_K))
        results[eid] = {
            "query": q,
            "category": entity["category"],
            "oracle_size": len(oracle),
            "oracle_ids": sorted(oracle),
            "backends": {},
        }
        for backend in BACKENDS:
            top = backend_topk(backend, q, k=K)
            hits = set(top) & oracle
            results[eid]["backends"][backend] = {
                "top_k_ids": top,
                "hits": sorted(hits),
                "recall_at_k": len(hits) / len(oracle) if oracle else 0.0,
            }
        print(
            f"[{eid:20s}] oracle={len(oracle):3d}  "
            f"V1={results[eid]['backends']['gemini-embedding-001']['recall_at_k']:.3f}  "
            f"V2={results[eid]['backends']['gemini-embedding-2']['recall_at_k']:.3f}  "
            f"Qwen={results[eid]['backends']['qwen3-embedding-8b-fp8-vllm']['recall_at_k']:.3f}"
        )

    out_path = OUT / f"entity_probe_results_oracle{ORACLE_K}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")

    # ------ Aggregate stats ------
    print("\n=== Aggregate: mean recall@{} per backend ===".format(K))
    per_backend_mean = {b: 0.0 for b in BACKENDS}
    per_backend_count = {b: 0 for b in BACKENDS}
    per_category: dict[str, dict] = {}
    for eid, r in results.items():
        cat = r["category"]
        per_category.setdefault(cat, {b: [] for b in BACKENDS})
        for b in BACKENDS:
            val = r["backends"][b]["recall_at_k"]
            per_backend_mean[b] += val
            per_backend_count[b] += 1
            per_category[cat][b].append(val)
    for b in BACKENDS:
        if per_backend_count[b]:
            per_backend_mean[b] /= per_backend_count[b]
    print("Overall mean recall@{}:".format(K))
    for b, v in per_backend_mean.items():
        print(f"  {b:35s}  {v:.3f}")

    print("\n=== By category (mean recall@{}) ===".format(K))
    print(f"{'category':<15s} {'V1':>7s} {'V2':>7s} {'Qwen':>7s}  n")
    for cat, by_b in per_category.items():
        row = f"{cat:<15s} "
        for b in BACKENDS:
            vals = by_b[b]
            row += f"{np.mean(vals):>7.3f} "
        row += f"  {len(by_b[BACKENDS[0]])}"
        print(row)

    summary = {
        "k": K,
        "n_entities": len(ENTITIES),
        "overall_mean_recall": per_backend_mean,
        "by_category": {
            cat: {b: float(np.mean(vals)) for b, vals in by_b.items()}
            for cat, by_b in per_category.items()
        },
    }
    summary_path = OUT / f"entity_probe_summary_oracle{ORACLE_K}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
