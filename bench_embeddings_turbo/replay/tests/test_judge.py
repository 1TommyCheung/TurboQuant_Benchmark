"""Tests for judge_all_turns.py — prompt building + JSON parsing + majority vote."""
from __future__ import annotations
import json


def test_build_judge_prompt_includes_user_query_and_chunks():
    from replay.runners.judge_all_turns import build_judge_prompt
    user_text = "find the latest email from yuqi"
    gold_facts = ["Tommy's address is 10 Shanghai Road since 2025-06-01"]
    gemini_chunks = [{"chunk_id": "g-1", "source_type": "email", "chunk_text": "From: yuqi...", "date_sgt": "2026-03-31", "party_from": "gjc"}]
    harrier_chunks = [{"chunk_id": "h-1", "source_type": "email", "chunk_text": "From: yuqi...", "date_sgt": "2026-03-31", "party_from": "gjc"}]
    prompt = build_judge_prompt(user_text, gold_facts, gemini_chunks, harrier_chunks)
    assert "find the latest email from yuqi" in prompt
    assert "10 Shanghai Road" in prompt
    assert "From: yuqi" in prompt
    for v in ("sufficient", "partially_sufficient", "insufficient", "better_than_gemini"):
        assert v in prompt


def test_parse_judge_response_extracts_4_bucket_verdict():
    from replay.runners.judge_all_turns import parse_judge_response
    raw = json.dumps({
        "harrier": {"verdict": "sufficient", "rationale": "All present.", "missing_evidence": "", "extra_evidence": ""},
        "gemini":  {"verdict": "sufficient", "rationale": "OK."},
    })
    parsed = parse_judge_response(raw)
    assert parsed["harrier"]["verdict"] == "sufficient"
    assert parsed["gemini"]["verdict"] == "sufficient"


def test_parse_judge_response_handles_markdown_codeblock():
    from replay.runners.judge_all_turns import parse_judge_response
    raw = "```json\n" + json.dumps({
        "harrier": {"verdict": "insufficient", "rationale": "Missing letter."},
        "gemini":  {"verdict": "sufficient", "rationale": "OK."},
    }) + "\n```"
    parsed = parse_judge_response(raw)
    assert parsed["harrier"]["verdict"] == "insufficient"


def test_majority_verdict_3_retries():
    from replay.runners.judge_all_turns import majority_verdict
    assert majority_verdict(["sufficient", "sufficient", "partially_sufficient"]) == ("sufficient", True)


def test_majority_verdict_split_decision():
    from replay.runners.judge_all_turns import majority_verdict
    v, consistent = majority_verdict(["sufficient", "insufficient", "partially_sufficient"])
    assert consistent is False
