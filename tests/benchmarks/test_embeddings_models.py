from __future__ import annotations
import pytest
from tqbench.benchmarks.embeddings.models import load_registry, get_candidate, ModelSpec, baseline_dim


def test_registry_loads():
    reg = load_registry()
    assert len(reg) > 0
    assert all(isinstance(m, ModelSpec) for m in reg)


def test_every_candidate_has_server_ref():
    for m in load_registry():
        assert m.server, f"Model '{m.id}' missing server reference"


def test_get_candidate_by_id():
    spec = get_candidate("qwen3-embedding-8b-q8-ollama")
    assert spec.dim == 4096
    assert spec.server == "ollama-local"


def test_get_candidate_missing_raises():
    with pytest.raises(KeyError):
        get_candidate("does-not-exist")


def test_baseline_dim():
    assert baseline_dim() == 3072
