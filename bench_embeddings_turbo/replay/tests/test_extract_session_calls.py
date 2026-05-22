"""Tests for extract_session_calls.py — parse the pi-session HTML into per-call records."""
from __future__ import annotations
import json
import pytest


def test_extract_calls_returns_list_of_records(sample_session_html, tmp_path):
    from replay.runners.extract_session_calls import extract_session_calls

    if not sample_session_html.exists():
        pytest.skip("sample session HTML not present")

    out_path = tmp_path / "session_calls.json"
    calls = extract_session_calls(sample_session_html, out_path)

    assert 25 <= len(calls) <= 35
    for c in calls:
        assert "turn_idx" in c
        assert "tool_name" in c
        assert c["tool_name"] in ("search_evidence", "build_evidence_pack")
        assert "args" in c
        assert "returned_chunk_ids" in c
        assert isinstance(c["returned_chunk_ids"], list)
    assert out_path.exists()


def test_extract_includes_user_turns(sample_session_html, tmp_path):
    from replay.runners.extract_session_calls import extract_session_calls

    if not sample_session_html.exists():
        pytest.skip("sample session HTML not present")

    out_path = tmp_path / "session_calls.json"
    calls = extract_session_calls(sample_session_html, out_path)
    user_turns = {c["turn_idx"]: c["user_text"] for c in calls if "user_text" in c}
    assert 1 in user_turns
    assert "yuqi" in user_turns[1].lower() or "lee" in user_turns[1].lower()


def test_cited_evidence_ids_per_turn(sample_session_html, tmp_path):
    from replay.runners.extract_session_calls import extract_cited_per_turn

    if not sample_session_html.exists():
        pytest.skip("sample session HTML not present")

    cited = extract_cited_per_turn(sample_session_html)
    assert isinstance(cited, dict)
    assert any(len(v) > 0 for v in cited.values())
