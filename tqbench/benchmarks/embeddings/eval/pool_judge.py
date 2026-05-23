"""Pool-and-judge for Layer 1 ground truth.

For each query:
1. Take union of top-20 chunk_ids across all candidates + Gemini.
2. Send (query, chunk_text) pairs to Claude Sonnet for graded relevance (0-3).
3. Output: qrels keyed by chunk_id → graded score.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from litellm import completion

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are evaluating whether a retrieved evidence chunk is relevant to a Singapore Family Court litigation search query. Rate the chunk's relevance on a 0-3 scale:

- 3 = Highly relevant: directly answers the query.
- 2 = Relevant: contains specific information related to the query.
- 1 = Marginally relevant: tangentially related but does not directly inform.
- 0 = Not relevant.

Query: "{query}"
Filters applied: {filters}

Evidence chunk (source_type={source_type}, party={party}, date={date}):
\"\"\"
{chunk_text}
\"\"\"

Return ONLY a single digit (0, 1, 2, or 3). No explanation."""


def judge_pair(
    query: str,
    chunk: dict[str, Any],
    filters: dict[str, Any] | None = None,
    model: str = "anthropic/claude-sonnet-4-6",
    seed: int = 42,
) -> int:
    """Return a graded relevance score 0-3 for (query, chunk)."""
    prompt = JUDGE_PROMPT.format(
        query=query,
        filters=json.dumps(filters or {}),
        source_type=chunk.get("source_type", "?"),
        party=chunk.get("party_from", "?"),
        date=chunk.get("date_sgt", "?"),
        chunk_text=chunk.get("chunk_text", "")[:2000],
    )
    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=seed,
            max_tokens=4,
        )
        txt = resp.choices[0].message.content.strip()
        for c in txt:
            if c in "0123":
                return int(c)
        log.warning(f"Unparseable judgment: {txt!r}")
        return 0
    except Exception as e:
        log.warning(f"Judge failed: {e}")
        return 0
