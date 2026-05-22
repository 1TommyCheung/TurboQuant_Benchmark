"""Tests for decision rule v2: weighted scoring + hard vetoes + non-decision branch."""
from __future__ import annotations
import pytest
from bench.scoring import (
    Verdict, CandidateScore, ModelResult,
    apply_vetoes, weighted_total, decide,
)


def _make_result(
    model_id: str = "qwen3-int8",
    quality_vector: float = 70.0,
    quality_e2e: float = 75.0,
    long_ctx: float = 80.0,
    local_control: float = 60.0,
    e2e_recall_by_source: dict[str, float] | None = None,
    dim: int = 4096,
    baseline_dim: int = 3072,
    gemini_e2e_recall_by_source: dict[str, float] | None = None,
) -> ModelResult:
    e2e = e2e_recall_by_source or {"court_doc": 75.0, "solicitor_letter": 78.0, "email": 70.0}
    base = gemini_e2e_recall_by_source or {"court_doc": 76.0, "solicitor_letter": 80.0, "email": 68.0}
    return ModelResult(
        model_id=model_id,
        quality_vector_only=quality_vector,
        quality_end_to_end=quality_e2e,
        long_context=long_ctx,
        local_control=local_control,
        e2e_recall_by_source_type=e2e,
        baseline_e2e_recall_by_source_type=base,
        dim=dim,
        baseline_dim=baseline_dim,
    )


def test_weighted_total_components():
    # Pin dim==baseline_dim so this test isolates the weighting components
    # (the dim penalty is exercised separately in test_dim_penalty_*).
    r = _make_result(quality_vector=80, quality_e2e=70, long_ctx=60, local_control=40,
                     dim=3072, baseline_dim=3072)
    # 0.15*80 + 0.35*70 + 0.25*60 + 0.25*40 = 12 + 24.5 + 15 + 10 = 61.5
    assert abs(weighted_total(r) - 61.5) < 0.001


def test_apply_vetoes_passes_when_within_floor():
    r = _make_result(
        e2e_recall_by_source={"court_doc": 73.0, "solicitor_letter": 78.0},
        gemini_e2e_recall_by_source={"court_doc": 76.0, "solicitor_letter": 80.0},
    )
    vetoes = apply_vetoes(r)
    assert vetoes == []


def test_apply_vetoes_fires_on_court_doc_regression():
    r = _make_result(
        e2e_recall_by_source={"court_doc": 68.0, "solicitor_letter": 78.0},  # -8 pts
        gemini_e2e_recall_by_source={"court_doc": 76.0, "solicitor_letter": 80.0},
    )
    vetoes = apply_vetoes(r)
    assert len(vetoes) == 1
    assert "court_doc" in vetoes[0]


def test_apply_vetoes_fires_on_solicitor_letter():
    r = _make_result(
        e2e_recall_by_source={"court_doc": 75.0, "solicitor_letter": 70.0},  # -10 pts
        gemini_e2e_recall_by_source={"court_doc": 76.0, "solicitor_letter": 80.0},
    )
    vetoes = apply_vetoes(r)
    assert any("solicitor_letter" in v for v in vetoes)


def test_dim_penalty_applied_when_different():
    r = _make_result(dim=4096, baseline_dim=3072,
                     quality_vector=80, quality_e2e=70, long_ctx=60, local_control=40)
    # Base = 61.5, penalty -3 → 58.5
    assert abs(weighted_total(r) - 58.5) < 0.001


def test_dim_penalty_zero_when_matched():
    r = _make_result(dim=3072, baseline_dim=3072,
                     quality_vector=80, quality_e2e=70, long_ctx=60, local_control=40)
    assert abs(weighted_total(r) - 61.5) < 0.001


def test_decide_switch_when_clear_winner():
    candidate = _make_result(model_id="qwen3-int8",
                             quality_vector=75, quality_e2e=78, long_ctx=82, local_control=70,
                             e2e_recall_by_source={"court_doc": 78, "solicitor_letter": 82})
    baseline = _make_result(model_id="gemini",
                            quality_vector=65, quality_e2e=68, long_ctx=55, local_control=10,
                            dim=3072,
                            e2e_recall_by_source={"court_doc": 76, "solicitor_letter": 80},
                            gemini_e2e_recall_by_source={"court_doc": 76, "solicitor_letter": 80})
    verdict = decide(candidate, baseline, all_candidates=[candidate, baseline])
    assert verdict.verdict == "switch"
    assert verdict.winner_id == "qwen3-int8"


def test_decide_stay_when_within_3pts():
    candidate = _make_result(model_id="qwen3-int8",
                             quality_vector=70, quality_e2e=70, long_ctx=70, local_control=70,
                             e2e_recall_by_source={"court_doc": 75, "solicitor_letter": 78})
    baseline = _make_result(model_id="gemini", dim=3072,
                            quality_vector=72, quality_e2e=70, long_ctx=70, local_control=68,
                            e2e_recall_by_source={"court_doc": 76, "solicitor_letter": 80},
                            gemini_e2e_recall_by_source={"court_doc": 76, "solicitor_letter": 80})
    verdict = decide(candidate, baseline, all_candidates=[candidate, baseline])
    assert verdict.verdict == "stay"


def test_decide_stay_when_veto_fires():
    candidate = _make_result(model_id="qwen3-int8",
                             quality_vector=90, quality_e2e=85, long_ctx=80, local_control=80,
                             e2e_recall_by_source={"court_doc": 65, "solicitor_letter": 82})  # -11 vs 76
    baseline = _make_result(model_id="gemini", dim=3072,
                            quality_vector=65, quality_e2e=68, long_ctx=55, local_control=10,
                            e2e_recall_by_source={"court_doc": 76, "solicitor_letter": 80},
                            gemini_e2e_recall_by_source={"court_doc": 76, "solicitor_letter": 80})
    verdict = decide(candidate, baseline, all_candidates=[candidate, baseline])
    assert verdict.verdict == "stay"
    assert any("court_doc" in v for v in verdict.veto_reasons)
