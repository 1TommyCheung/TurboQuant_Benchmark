"""Source-type weights for cited-evidence overlap in Phase A.

Per spec §5.3 / §5.4 — legal documents weighted higher than conversational
evidence because missing a solicitor letter or court order is far more
consequential than missing a single WhatsApp ping.
"""
from __future__ import annotations

SOURCE_TYPE_WEIGHTS: dict[str, float] = {
    "solicitor_letter":  2.0,
    "court_doc":         2.0,
    "email":             1.5,
    "document_exchange": 1.5,
    "financial":         1.5,
    "whatsapp":          1.0,
    "photo":             1.0,
}

DEFAULT_WEIGHT: float = 1.0


def weight_for(source_type: str) -> float:
    return SOURCE_TYPE_WEIGHTS.get(source_type, DEFAULT_WEIGHT)


def weighted_cited_overlap(
    cited_with_type: list[tuple[str, str]],
    returned_chunk_ids: set[str],
) -> float:
    """Source-type-weighted fraction of cited chunks present in the backend's
    returned top-K. Returns 1.0 for no-op turns (no cited chunks)."""
    if not cited_with_type:
        return 1.0
    total_weight = 0.0
    matched_weight = 0.0
    for cid, st in cited_with_type:
        w = weight_for(st)
        total_weight += w
        if cid in returned_chunk_ids:
            matched_weight += w
    if total_weight == 0:
        return 1.0
    return matched_weight / total_weight
