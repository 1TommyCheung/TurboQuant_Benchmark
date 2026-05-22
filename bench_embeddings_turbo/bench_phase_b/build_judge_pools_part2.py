"""Build judge prompt pools for the REMAINING 10 semantic queries (Part 2 of pool-and-judge)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from build_judge_pools import (
    BACKENDS, vector_topk, bm25_topk, build_prompt,
    SONNET_INTRO, OPUS_INTRO, SKEPTICAL_INTRO,
)

import json
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bench.io_lance import read_prod_chunks
from bench.snapshot import SNAPSHOT_SEARCH_DUCKDB

OUT = Path("/tmp/judge_pools")
OUT.mkdir(exist_ok=True)

REMAINING_QUERIES = [
    {"id": "valley_point_handover", "category": "event_description",
     "query": "Valley Point pick up drop off children",
     "intent": "Find chunks about handover events near Valley Point Shopping Centre"},
    {"id": "negative_characterisation", "category": "concept",
     "query": "negative characterisation father children alienation influence",
     "intent": "Find chunks describing a behavioural pattern of negative portrayal of the father"},
    {"id": "children_refuse_visit", "category": "synonym_paraphrase",
     "query": "children don't want to see father refuse visit access",
     "intent": "Find chunks where children expressed reluctance to see father, however phrased"},
    {"id": "tracy_limited_access_proposals", "category": "topical",
     "query": "limited meal only access proposals suspended children",
     "intent": "Find proposals that constrained access regardless of specific words used"},
    {"id": "hk_unilateral_travel", "category": "event_description",
     "query": "Hong Kong children taken without consent passports missing",
     "intent": "Find narratives of the Dec 2024 Hong Kong incident"},
    {"id": "father_school_visit_framing", "category": "concept",
     "query": "father visit school kindergarten approaching children negative framing harassment",
     "intent": "Find chunks discussing legal characterisation of school visits"},
    {"id": "counselling_engagement_pattern", "category": "topical",
     "query": "counselling sessions in-person tele-conference engagement willingness refusal",
     "intent": "Find chunks about engagement pattern with counselling"},
    {"id": "gatekeeping_pattern", "category": "concept",
     "query": "gatekeeping behaviour mother exclude father co-parenting interference",
     "intent": "Find chunks about gatekeeping legal concept and its manifestations"},
    {"id": "fdr_mediation_directions", "category": "topical",
     "query": "FDR mediation directions counsellor third party suspension access",
     "intent": "Find chunks describing the mediation process and its outputs"},
    {"id": "matrimonial_home_valuation", "category": "topical",
     "query": "matrimonial home valuation arms-length sale market value division",
     "intent": "Find chunks about valuation methodology"},
]


def main() -> None:
    con = duckdb.connect(str(SNAPSHOT_SEARCH_DUCKDB), read_only=True)
    df = read_prod_chunks()
    df = df.set_index("chunk_id")
    lookup = df.to_dict("index")
    print(f"Loaded {len(lookup)} chunks")

    manifest = {}
    for q in REMAINING_QUERIES:
        qid = q["id"]
        print(f"\n=== {qid} ===")
        pool: set[str] = set()
        for backend in BACKENDS:
            top = vector_topk(backend, q["query"], k=20)
            pool.update(top)
        bm25 = bm25_topk(con, q["query"], k=20)
        pool.update(bm25)
        print(f"  Pool size: {len(pool)}")

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

    manifest_path = OUT / "_manifest_part2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {manifest_path}")
    print(f"Total new prompts: {len(REMAINING_QUERIES)} × 3 = {len(REMAINING_QUERIES) * 3}")


if __name__ == "__main__":
    main()
