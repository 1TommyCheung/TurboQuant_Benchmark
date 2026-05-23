"""Mine agent_verified_facts.jsonl for Gemini-failure cases.

A fact is a "Gemini failure" if:
- It was superseded (tombstoned with `supersedes` field)
- OR its `confidence` field (when present) is below a threshold

For each such fact, the manually-identified "right" evidence_ids serve
as positives. The fact's `category` and key claim phrases form a query.

Output: data/eval_queries/adversarial_gemini_failures.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from tqbench.benchmarks.embeddings.eval.snapshot import SNAPSHOT_FACTS_JSONL

# Frozen snapshot — see eval/snapshot.py + SNAPSHOT.md.
FACTS_PATH = SNAPSHOT_FACTS_JSONL
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_queries" / "adversarial_gemini_failures.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _fact_to_query(fact: dict) -> str:
    """Extract a search-style query from a fact record."""
    parts: list[str] = []
    if fact.get("category"):
        parts.append(fact["category"])
    text = fact.get("fact", "") or fact.get("claim", "")
    # Take first 25 words as query approximation
    parts.append(" ".join(text.split()[:25]))
    return " ".join(parts).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts-path", type=Path, default=FACTS_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if not args.facts_path.exists():
        log.warning(f"No facts file at {args.facts_path}; writing empty file.")
        args.out.write_text("")
        return

    out: list[dict] = []
    for i, line in enumerate(args.facts_path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            fact = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Filter to Gemini-failure cases: superseded OR low-confidence
        is_failure = (
            bool(fact.get("supersedes"))
            or (fact.get("confidence") is not None and fact["confidence"] < 0.7)
        )
        if not is_failure:
            continue
        query = _fact_to_query(fact)
        if not query:
            continue
        out.append({
            "id": f"ADV-{i:04d}",
            "query": query,
            "positives": fact.get("evidence_ids", []),
            "category": fact.get("category"),
            "fact_id": fact.get("id"),
            "rationale": "supersedes" if fact.get("supersedes") else "low_confidence",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    log.info(f"Wrote {len(out)} adversarial queries to {args.out}")


if __name__ == "__main__":
    main()
