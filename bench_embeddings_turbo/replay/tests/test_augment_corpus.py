"""Tests for augment_corpus.py — atomic LanceDB augmentation from snapshot."""
from __future__ import annotations
import pytest


def test_extract_chunk_ids_from_html_pi_session(sample_session_html):
    """Parse the pi-session HTML and extract every chunk_id mentioned in tool results."""
    from replay.runners.augment_corpus import extract_referenced_chunk_ids

    if not sample_session_html.exists():
        pytest.skip("sample session HTML not present in this environment")

    chunk_ids = extract_referenced_chunk_ids(sample_session_html)
    assert isinstance(chunk_ids, set)
    assert 50 < len(chunk_ids) < 1000, f"unexpected chunk count: {len(chunk_ids)}"
    for cid in chunk_ids:
        assert isinstance(cid, str)
        assert len(cid) > 8


def test_extract_chunk_ids_handles_evidence_ids_in_save_case_fact():
    from replay.runners.augment_corpus import _extract_evidence_ids_from_save_case_fact_args
    args = {"category": "property", "evidence_ids": ["id-1", "id-2", "id-3"]}
    out = _extract_evidence_ids_from_save_case_fact_args(args)
    assert out == {"id-1", "id-2", "id-3"}


def test_extract_chunk_ids_inline_assistant_text():
    from replay.runners.augment_corpus import _extract_inline_evidence_ids
    text = "Per [5e6bff96-f72d-428f-b0bc-3f7d5b46b3ea] the address is..."
    out = _extract_inline_evidence_ids(text)
    assert "5e6bff96-f72d-428f-b0bc-3f7d5b46b3ea" in out


def test_extract_chunk_ids_rfc2822_message_id():
    from replay.runners.augment_corpus import _extract_inline_evidence_ids
    text = "See email [007e01dcc0c1$2d22aaa0$8767ffe0$@gjclaw.com.sg] from gjc"
    out = _extract_inline_evidence_ids(text)
    assert "007e01dcc0c1$2d22aaa0$8767ffe0$@gjclaw.com.sg" in out


def test_compute_augment_set_filters_correctly():
    """Augment set = referenced chunk_ids - already-in-bench chunk_ids."""
    from replay.runners.augment_corpus import compute_augment_set
    referenced_chunk_ids = {"a", "c", "e"}   # already in snapshot's chunk_id space
    bench_chunk_ids = {"a", "b"}              # already in bench
    augment_set = compute_augment_set(referenced_chunk_ids, bench_chunk_ids)
    assert augment_set == {"c", "e"}
