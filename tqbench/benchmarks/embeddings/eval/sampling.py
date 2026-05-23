"""Stratified 50K-chunk corpus sampler from production LanceDB.

Per spec §7: boost legally critical source_types so per-stratum
statistical power is adequate for the hard vetoes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# Quotas reflect actual production supply (252,629 chunks total as of 2026-05).
#   solicitor_letter capped at 2,400 (supply ~2,455) — legally critical but rare.
#   financial capped at 300 (supply ~342) — same.
# The shortfall was reallocated to court_doc (the other floor-protected source
# under spec §3) and to high-volume email + document_exchange. Total = 50,000.
SAMPLE_QUOTAS: dict[str, int] = {
    "email": 18_500,
    "whatsapp": 18_000,
    "court_doc": 7_500,
    "solicitor_letter": 2_400,   # capped to ~2455 supply
    "photo_video": 2_000,         # photo + video collapsed
    "document_exchange": 1_300,
    "financial": 300,             # capped to ~342 supply
}

PHOTO_VIDEO_TYPES = frozenset({"photo", "video"})


def length_bucket(token_count: int) -> str:
    """Map token count to one of {short, medium, long, very_long}."""
    if token_count < 512:
        return "short"
    if token_count < 2048:
        return "medium"
    if token_count < 8192:
        return "long"
    return "very_long"


def stratified_sample(chunks: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Draw a stratified 50K sample with per-source-type quotas from SAMPLE_QUOTAS.

    photo and video chunks are pooled into a single 'photo_video' stratum.
    Adds a 'length_bucket' column derived from token_count.
    Raises ValueError if any stratum has insufficient supply for its quota.
    """
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []

    for stratum_name, quota in SAMPLE_QUOTAS.items():
        if stratum_name == "photo_video":
            pool = chunks[chunks["source_type"].isin(PHOTO_VIDEO_TYPES)]
        else:
            pool = chunks[chunks["source_type"] == stratum_name]

        if len(pool) < quota:
            raise ValueError(
                f"Stratum '{stratum_name}' has insufficient data: "
                f"have {len(pool)}, need {quota}"
            )

        idx = rng.choice(pool.index.values, size=quota, replace=False)
        pieces.append(pool.loc[idx])

    sample = pd.concat(pieces, ignore_index=True)
    sample["length_bucket"] = sample["token_count"].apply(length_bucket)
    return sample
