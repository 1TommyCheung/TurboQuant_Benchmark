from __future__ import annotations
import pytest
from tqbench.benchmarks.generation.models import (
    load_registry, get_candidate, ModelSpec, quality_groups,
)


def test_registry_loads_10_candidates():
    reg = load_registry()
    assert len(reg) == 10
    assert all(isinstance(m, ModelSpec) for m in reg)


def test_every_candidate_has_server_and_model_name():
    for m in load_registry():
        assert m.server, f"'{m.id}' missing server"
        assert m.model_name, f"'{m.id}' missing model_name"


def test_get_candidate_by_id():
    spec = get_candidate("qwen3.5-9b-fp8-vllm")
    assert spec.server == "vllm-docker"
    assert spec.model_name == "lovedheart/Qwen3.5-9B-FP8"
    assert spec.spec_decode is None
    assert spec.quality_group == "fp8"


def test_get_candidate_with_spec_decode():
    spec = get_candidate("qwen3.5-9b-fp8-vllm-dflash")
    assert spec.spec_decode == "dflash"
    assert spec.drafter_repo == "z-lab/Qwen3.5-9B-DFlash"


def test_get_candidate_with_pflash():
    spec = get_candidate("qwen3.5-9b-q8-lucebox-pflash")
    assert spec.spec_decode == "dflash"
    assert spec.spec_prefill == "pflash"


def test_get_candidate_missing_raises():
    with pytest.raises(KeyError):
        get_candidate("does-not-exist")


def test_quality_groups():
    groups = quality_groups()
    assert "fp8" in groups
    assert "q8" in groups
    assert len(groups["fp8"]) >= 1
    assert len(groups["q8"]) >= 1
    for gname, specs in groups.items():
        for s in specs:
            assert s.quality_group == gname
