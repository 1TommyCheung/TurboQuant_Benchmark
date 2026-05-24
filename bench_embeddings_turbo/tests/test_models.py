"""Tests for model registry + config loading."""
from __future__ import annotations
import pytest
from bench.models import load_registry, get_candidate, ModelSpec


def test_registry_has_19_candidates():
    """Round 1 added 6 candidates; Round 2 (2026-05-16) added 4 more
    (Qwen3-4B FP16, Jina v5 small, Qwen3-8B Q8 via Ollama, Qwen3-8B FP8 via vLLM)."""
    reg = load_registry()
    assert len(reg) == 19


def test_baseline_is_gemini():
    reg = load_registry()
    g = next(c for c in reg if c.id == "gemini-embedding-001")
    assert g.kind == "api"
    assert g.dim == 3072


def test_get_candidate_by_id():
    spec = get_candidate("qwen3-embedding-8b-int8")
    assert spec.dim == 4096
    assert spec.precision == "int8"


def test_get_candidate_missing_raises():
    with pytest.raises(KeyError):
        get_candidate("does-not-exist")


def test_spec_has_required_fields():
    spec = get_candidate("harrier-oss-0.6b-bf16")
    assert spec.id and spec.dim and spec.max_ctx_tokens and spec.precision
