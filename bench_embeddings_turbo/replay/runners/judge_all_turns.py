"""Phase A: Sonnet judges all 11 turns using the 4-bucket scale (spec §5.5).

Snapshot-aligned: pulls chunk text from SNAPSHOT_LANCEDB_PATH and gold facts
from SNAPSHOT_FACTS_JSONL.

Writes: bench_embeddings/reports/raw/{date}_phase_a_judge.json
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bench.snapshot import (
    SNAPSHOT_LANCEDB_PATH, SNAPSHOT_FACTS_JSONL, assert_snapshot_present
)
from litellm import completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BENCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_DIFF = BENCH_ROOT / "reports" / "raw" / f"{dt.date.today().isoformat()}_phase_a_static_diff.json"
DEFAULT_OUT = BENCH_ROOT / "reports" / "raw" / f"{dt.date.today().isoformat()}_phase_a_judge.json"

SONNET_MODEL = "anthropic/claude-sonnet-4-6"
N_RETRIES = 3


def _load_facts() -> list[dict]:
    if not SNAPSHOT_FACTS_JSONL.exists():
        return []
    out = []
    for line in SNAPSHOT_FACTS_JSONL.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _facts_for_query(facts: list[dict], query: str) -> list[str]:
    out = []
    q_tokens = set(re.findall(r"\w+", query.lower())) - {"the", "a", "an", "and", "or", "of", "in"}
    for f in facts:
        ftext = (f.get("fact") or f.get("claim") or "").lower()
        if not ftext:
            continue
        f_tokens = set(re.findall(r"\w+", ftext))
        if len(q_tokens & f_tokens) >= 2:
            out.append(f.get("fact") or f.get("claim"))
    return out[:5]


def build_judge_prompt(
    user_text: str,
    gold_facts: list[str],
    gemini_chunks: list[dict],
    harrier_chunks: list[dict],
) -> str:
    def fmt(chunks: list[dict]) -> str:
        lines = []
        for c in chunks:
            lines.append(
                f"[chunk_id={c.get('chunk_id')} source_type={c.get('source_type')} "
                f"date={c.get('date_sgt')} party={c.get('party_from')}]\n"
                f"{(c.get('chunk_text') or '')[:600]}"
            )
        return "\n\n".join(lines) or "(empty)"

    facts_section = "\n".join(f"- {f}" for f in gold_facts) if gold_facts else "(none directly relevant)"

    return f"""You are evaluating two retrieval backends for an evidence-research agent
serving a self-represented litigant (Tommy Cheung) in Singapore Family Court
case FC/OAD 22/2025.

USER QUERY (verbatim from the pi-session):
{user_text!r}

GOLD VERIFIED FACTS (any agent_verified_facts.jsonl entries touching this query):
{facts_section}

CHUNKS RETURNED BY GEMINI (top-10 across all search_evidence calls in this turn):
{fmt(gemini_chunks)}

CHUNKS RETURNED BY HARRIER:
{fmt(harrier_chunks)}

TASK: Would a careful Singapore Family Court litigant find each backend's chunks
adequate to draft the same affidavit / letter / position Tommy actually drafted
in the original session?

Score EACH BACKEND on this 4-bucket scale:
- "sufficient" - all load-bearing evidence is present; the agent could draft the same conclusion
- "partially_sufficient" - main argument supported, but some supporting detail missing; the agent would have to retry
- "insufficient" - load-bearing evidence missing; the agent's draft would be materially weaker
- "better_than_gemini" - this backend surfaces something the other missed that is actually relevant (Harrier only; Gemini cannot self-score this)

Output JSON ONLY, no other text:
{{
  "harrier": {{
    "verdict": "sufficient" | "partially_sufficient" | "insufficient" | "better_than_gemini",
    "rationale": "one-sentence reason",
    "missing_evidence": "what Harrier did not surface that Gemini did and that matters (empty if none)",
    "extra_evidence": "what Harrier surfaced that Gemini did not and is relevant (empty if none)"
  }},
  "gemini": {{
    "verdict": "sufficient" | "partially_sufficient" | "insufficient",
    "rationale": "one-sentence reason"
  }}
}}
"""


def parse_judge_response(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def majority_verdict(verdicts: list[str]) -> tuple[str, bool]:
    cnt = Counter(verdicts)
    top, top_n = cnt.most_common(1)[0]
    return top, top_n > len(verdicts) / 2


def _call_sonnet(prompt: str, seed: int) -> dict | None:
    try:
        resp = completion(
            model=SONNET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
            seed=seed,
        )
        return parse_judge_response(resp.choices[0].message.content)
    except Exception as e:
        log.warning(f"  Sonnet call failed: {e}")
        return None


def _chunks_for_turn(
    static_diff_rows: list[dict],
    turn_idx: int,
    backend_id: str,
    chunk_text_lookup: dict[str, dict],
    k: int = 10,
) -> list[dict]:
    seen: set[str] = set()
    chunks: list[dict] = []
    for row in static_diff_rows:
        if row["turn_idx"] != turn_idx:
            continue
        for cid in row["backends"][backend_id]["e2e_top_k"]:
            if cid in seen:
                continue
            seen.add(cid)
            chunk = chunk_text_lookup.get(cid)
            if chunk:
                chunks.append({
                    "chunk_id": cid,
                    "source_type": chunk.get("source_type", ""),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "date_sgt": chunk.get("date_sgt", ""),
                    "party_from": chunk.get("party_from", ""),
                })
            if len(chunks) >= k:
                break
        if len(chunks) >= k:
            break
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-diff", type=Path, default=DEFAULT_STATIC_DIFF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    assert_snapshot_present()

    log.info(f"Loading static diff from {args.static_diff}")
    diff = json.loads(args.static_diff.read_text())
    rows = diff["rows"]
    backends = diff["backends"]
    assert len(backends) >= 2

    log.info(f"Loading gold facts from snapshot {SNAPSHOT_FACTS_JSONL}")
    facts = _load_facts()
    log.info(f"  {len(facts)} facts available")

    log.info(f"Loading chunk text lookup from snapshot {SNAPSHOT_LANCEDB_PATH}")
    db_snap = lancedb.connect(str(SNAPSHOT_LANCEDB_PATH))
    snap_tbl = db_snap.open_table("chunks")
    chunk_text_lookup = (
        snap_tbl.to_pandas()[["chunk_id", "source_type", "chunk_text", "date_sgt", "party_from"]]
        .set_index("chunk_id")
        .to_dict("index")
    )

    turns = sorted({r["turn_idx"] for r in rows})
    out_turns = []
    for turn_idx in turns:
        log.info(f"Judging turn {turn_idx}/{len(turns)}...")
        turn_calls = [r for r in rows if r["turn_idx"] == turn_idx]
        user_text = turn_calls[0]["user_text"] if turn_calls else ""
        joined_q = " ".join(c["query"] for c in turn_calls)
        gold_facts = _facts_for_query(facts, joined_q + " " + (user_text or ""))

        gemini_chunks = _chunks_for_turn(rows, turn_idx, backends[0], chunk_text_lookup, k=10)
        harrier_chunks = _chunks_for_turn(rows, turn_idx, backends[1], chunk_text_lookup, k=10)
        prompt = build_judge_prompt(user_text, gold_facts, gemini_chunks, harrier_chunks)

        retries = []
        for r in range(N_RETRIES):
            v = _call_sonnet(prompt, seed=42 + r)
            if v:
                retries.append(v)

        harrier_verdicts = [r["harrier"]["verdict"] for r in retries if "harrier" in r]
        gemini_verdicts = [r["gemini"]["verdict"] for r in retries if "gemini" in r]
        harrier_v, harrier_ok = majority_verdict(harrier_verdicts) if harrier_verdicts else ("error", False)
        gemini_v, gemini_ok = majority_verdict(gemini_verdicts) if gemini_verdicts else ("error", False)

        out_turns.append({
            "turn_idx": turn_idx,
            "user_text": user_text,
            "gold_facts_used": gold_facts,
            "harrier": {"verdict_majority": harrier_v, "consistent": harrier_ok, "retries": retries},
            "gemini":  {"verdict_majority": gemini_v,  "consistent": gemini_ok},
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "date": dt.date.today().isoformat(),
        "model": SONNET_MODEL,
        "n_retries_per_turn": N_RETRIES,
        "turns": out_turns,
    }, indent=2, default=str))
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
