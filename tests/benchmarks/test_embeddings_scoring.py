from __future__ import annotations
from tqbench.benchmarks.embeddings.eval.scoring import (
    ModelResult, apply_vetoes, weighted_total, decide,
)


def _make_result(model_id: str, dim: int = 3072, e2e_court: float = 80.0,
                 base_court: float = 80.0, **kw) -> ModelResult:
    defaults = dict(
        quality_vector_only=70.0,
        quality_end_to_end=75.0,
        long_context=60.0,
        local_control=50.0,
        e2e_recall_by_source_type={"court_doc": e2e_court, "solicitor_letter": 85.0},
        baseline_e2e_recall_by_source_type={"court_doc": base_court, "solicitor_letter": 85.0},
        dim=dim,
        baseline_dim=3072,
    )
    defaults.update(kw)
    return ModelResult(model_id=model_id, **defaults)


def test_no_veto_when_within_threshold():
    r = _make_result("qwen", e2e_court=76.0, base_court=80.0)
    assert apply_vetoes(r) == []


def test_veto_when_regression_exceeds_threshold():
    r = _make_result("qwen", e2e_court=74.0, base_court=80.0)
    vetoes = apply_vetoes(r)
    assert len(vetoes) == 1
    assert "VETO" in vetoes[0]


def test_dim_penalty_applied():
    r = _make_result("qwen", dim=4096)
    total_with_penalty = weighted_total(r)
    r2 = _make_result("qwen", dim=3072)
    total_without = weighted_total(r2)
    assert total_with_penalty == total_without - 3.0


def test_decide_stay_when_all_vetoed():
    baseline = _make_result("gemini")
    candidate = _make_result("qwen", e2e_court=60.0, base_court=80.0)
    verdict = decide(candidate, baseline, [baseline, candidate])
    assert verdict.verdict == "stay"
