"""Build pool-and-judge prompt files for semantic-relevance scoring.

For each selected semantic query:
  1. Pool top-20 from V1/V2/Qwen3 vector + BM25 → unique chunk set
  2. Hydrate chunk_id → text snippet via the snapshot LanceDB
  3. Write 3 judge prompts (Sonnet, Opus, Skeptical-Opus) to disk

Each prompt asks the judge to score every pooled chunk on a 0-3 scale.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import duckdb
import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.io_lance import bench_lancedb_path, read_prod_chunks
from bench.models import load_embedder, get_candidate
from bench.snapshot import SNAPSHOT_SEARCH_DUCKDB


OUT = Path("/tmp/judge_pools")
OUT.mkdir(exist_ok=True)


SELECTED_QUERIES = [
    {"id": "parental_alienation", "category": "concept",
     "query": "parental alienation children refusing access negative perception father",
     "intent": "Find chunks discussing the legal/psychological concept of parental alienation in this case"},
    {"id": "access_denial_synonyms", "category": "synonym_paraphrase",
     "query": "access denied refused blocked cancelled children",
     "intent": "Find chunks about access denial regardless of specific word choice"},
    {"id": "valley_point_event", "category": "event_description",
     "query": "Valley Point Shopping Centre encounter Tristan November 2025",
     "intent": "Find narrative of a specific encounter/incident at or near Valley Point in Nov 2025"},
    {"id": "disclosure_non_compliance", "category": "topical",
     "query": "disclosure non-compliance failure produce documents adverse inference",
     "intent": "Find chunks about the disclosure dispute and non-compliance pattern"},
    {"id": "indirect_contribution_caregiver", "category": "concept",
     "query": "indirect contribution caregiver homemaker non-financial role sacrifice career",
     "intent": "Find chunks discussing indirect (non-financial) contribution under Women's Charter s.112"},
]


BACKENDS = ["gemini-embedding-001", "gemini-embedding-2", "qwen3-embedding-8b-fp8-vllm"]


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


# Judge prompt templates ----------------------------------------------------

RUBRIC = """## Scoring rubric
Score each chunk on this 0-3 scale for relevance to the user's query+intent:
- **0 = not relevant**: the chunk does not address the query topic at all
- **1 = weakly relevant**: tangentially related, mentions adjacent concepts but doesn't substantively address the query
- **2 = relevant**: substantively addresses the query topic; would be useful evidence for a Singapore family-court matter on this point
- **3 = highly relevant**: directly addresses the query; the kind of chunk a careful legal researcher would want surfaced first

When the chunk text is too short or generic to judge, score 0.
When a chunk is about a different case or a generic precedent that doesn't bear on Tommy Cheung v Tracy Cheuk, score at most 1.
"""

SONNET_INTRO = """You are a careful neutral evidence-relevance judge for Singapore Family Court case FC/OAD 22/2025 (Tommy Cheung v Tracy Cheuk).

You will score chunks for relevance to a semantic query. Be fair, calibrated, neither too lenient nor too strict.
"""

OPUS_INTRO = """You are a senior legal researcher (Opus) scoring evidence-chunk relevance for Singapore Family Court case FC/OAD 22/2025 (Tommy Cheung v Tracy Cheuk).

You will score chunks for relevance to a semantic query. Use careful legal reasoning. Consider both surface keywords AND the underlying legal/factual relevance.
"""

SKEPTICAL_INTRO = """You are an adversarial reviewer (Skeptical Opus) scoring evidence-chunk relevance for Singapore Family Court case FC/OAD 22/2025 (Tommy Cheung v Tracy Cheuk).

Your bias is to challenge over-broad relevance claims. For each chunk, actively try to argue why it is NOT relevant. Only score 2 or 3 if you genuinely cannot find a reason to disqualify the chunk. Be honest — if a chunk is clearly relevant despite your skepticism, score it accurately.
"""


def build_prompt(intro: str, query_obj: dict, pool_with_text: list[dict]) -> str:
    chunks_md = "\n".join(
        f"### Chunk {i+1} — chunk_id `{c['chunk_id']}`\n"
        f"- source_type: {c.get('source_type')}\n"
        f"- party_from: {c.get('party_from')}\n"
        f"- date: {c.get('date_sgt')}\n"
        f"- snippet:\n```\n{c.get('snippet', '')[:500]}\n```\n"
        for i, c in enumerate(pool_with_text)
    )
    output_schema = """```json
{
  "query_id": "<query_id>",
  "scores": [
    {"chunk_id": "<id>", "score": 0|1|2|3, "reason": "<one line>"}
  ]
}
```"""
    return f"""{intro}

## The query

- **Query string:** `{query_obj['query']}`
- **Intent:** {query_obj['intent']}
- **Category:** {query_obj['category']}

{RUBRIC}

## Chunks to score ({len(pool_with_text)} total)

{chunks_md}

## Output format

Return EXACTLY one JSON object at the end of your response (no other text after the closing fence):

{output_schema}

The `scores` array must contain one entry per chunk above, in the same order. Each entry: chunk_id (verbatim), score (integer 0-3), reason (one sentence explaining the score, citing the chunk's content).

Begin scoring now.
"""


def main() -> None:
    con = duckdb.connect(str(SNAPSHOT_SEARCH_DUCKDB), read_only=True)

    # Cache chunk lookup
    print("Loading chunk lookup from snapshot...")
    df = read_prod_chunks()
    df = df.set_index("chunk_id")
    lookup = df.to_dict("index")
    print(f"  loaded {len(lookup)} chunks")

    manifest = {}
    for q in SELECTED_QUERIES:
        qid = q["id"]
        print(f"\n=== {qid} ===")
        pool: set[str] = set()
        for backend in BACKENDS:
            top = vector_topk(backend, q["query"], k=20)
            pool.update(top)
            print(f"  {backend}: {len(top)} hits (running pool {len(pool)})")
        bm25 = bm25_topk(con, q["query"], k=20)
        pool.update(bm25)
        print(f"  bm25: {len(bm25)} hits (final pool {len(pool)})")

        pool_with_text = []
        for cid in sorted(pool):
            row = lookup.get(cid, {})
            pool_with_text.append({
                "chunk_id": cid,
                "source_type": row.get("source_type"),
                "party_from": row.get("party_from"),
                "date_sgt": str(row.get("date_sgt") or ""),
                "is_privileged": bool(row.get("is_privileged", False)),
                "snippet": (row.get("chunk_text") or "")[:500],
            })

        # Write three judge prompts
        for judge_label, intro in [
            ("sonnet", SONNET_INTRO),
            ("opus", OPUS_INTRO),
            ("skeptical_opus", SKEPTICAL_INTRO),
        ]:
            prompt = build_prompt(intro, q, pool_with_text)
            fname = f"{qid}__{judge_label}.prompt.txt"
            (OUT / fname).write_text(prompt)
            print(f"  wrote {fname} ({len(prompt)} chars)")
        manifest[qid] = {
            "query": q["query"],
            "category": q["category"],
            "pool_size": len(pool),
            "pool_chunk_ids": sorted(pool),
        }

    (OUT / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {OUT / '_manifest.json'}")
    print(f"Total prompts: {len(SELECTED_QUERIES)} queries × 3 judges = {len(SELECTED_QUERIES) * 3}")


if __name__ == "__main__":
    main()
