"""Pinned schema definitions — frozen at case_kb commit 1fe458f (2026-05-16).

These mirror the production `core/schemas.py` at the snapshot commit so the
benchmark continues to parse snapshot data correctly even if the upstream
schema evolves. If you need to update these, also rebuild the snapshot data
and bump SNAPSHOT_ID in bench/snapshot.py.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """Schema for chunks.parquet — retrieval-optimized text chunks.

    Pinned copy of core/schemas.py:ChunkRecord at case_kb commit 1fe458f.
    """
    chunk_id: str                          # uuid
    evidence_id: str                       # FK → evidence_master
    source_type: str                       # email, whatsapp, court_doc, etc.
    chunk_index: int = 0                   # position within parent document
    chunk_text: str = ""                   # the actual text chunk
    context_header: str = ""               # [Source: ... | Date: ... | Party: ...]
    token_count: int = 0                   # estimated token count
    output_eligible_for: list[str] = Field(default_factory=list)  # ["internal", "court", "court_costs"]
    party_from: str = "unknown"
    date_sgt: Optional[datetime] = None
    legal_issues: list[str] = Field(default_factory=list)
    is_privileged: bool = False
    is_wp: bool = False
    source_category: str = "direct_evidence"  # "direct_evidence" or "cited_precedent"
    pipeline_version: str = "0.1.0"
