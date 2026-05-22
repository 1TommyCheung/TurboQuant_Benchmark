"""Decision rule v2: weighted scoring + hard vetoes + non-decision branch.

Per spec §3:
- Quality vector-only 15%, end-to-end 35%, long-context 25%, local-control 25%.
- Hard veto: any candidate that regresses >5pts on court_doc or solicitor_letter
  end-to-end recall@10 vs Gemini baseline is disqualified.
- Dim penalty: -3pts if candidate dim != baseline dim.
- Non-decision: top-2 within 3pts AND every source_type within 5pts → stay.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# Weight constants
W_QUALITY_VECTOR: float = 0.15
W_QUALITY_E2E: float = 0.35
W_LONG_CONTEXT: float = 0.25
W_LOCAL_CONTROL: float = 0.25

VETO_SOURCE_TYPES: tuple[str, ...] = ("court_doc", "solicitor_letter")
VETO_THRESHOLD_PTS: float = 5.0
DIM_PENALTY_PTS: float = 3.0
NON_DECISION_TOTAL_TOL: float = 3.0
NON_DECISION_SOURCE_TOL: float = 5.0


@dataclass
class ModelResult:
    model_id: str
    quality_vector_only: float
    quality_end_to_end: float
    long_context: float
    local_control: float
    e2e_recall_by_source_type: dict[str, float]
    baseline_e2e_recall_by_source_type: dict[str, float]
    dim: int
    baseline_dim: int


@dataclass
class CandidateScore:
    model_id: str
    total: float
    veto_reasons: list[str] = field(default_factory=list)
    eligible: bool = True


@dataclass
class Verdict:
    verdict: str               # "switch" | "stay" | "inconclusive"
    winner_id: str | None
    scores: list[CandidateScore]
    veto_reasons: list[str]    # merged veto reasons from the winner
    rationale: str


def apply_vetoes(result: ModelResult) -> list[str]:
    """Return list of veto-reason strings; empty if candidate passes."""
    reasons: list[str] = []
    for st in VETO_SOURCE_TYPES:
        cand = result.e2e_recall_by_source_type.get(st)
        base = result.baseline_e2e_recall_by_source_type.get(st)
        if cand is None or base is None:
            continue
        if base - cand > VETO_THRESHOLD_PTS:
            reasons.append(
                f"VETO: {result.model_id} regresses {base - cand:.1f}pts on {st} "
                f"end-to-end recall@10 (>{VETO_THRESHOLD_PTS}pt floor)"
            )
    return reasons


def weighted_total(result: ModelResult) -> float:
    """Apply the 15/35/25/25 weighting + dim penalty."""
    total = (
        W_QUALITY_VECTOR * result.quality_vector_only
        + W_QUALITY_E2E * result.quality_end_to_end
        + W_LONG_CONTEXT * result.long_context
        + W_LOCAL_CONTROL * result.local_control
    )
    if result.dim != result.baseline_dim:
        total -= DIM_PENALTY_PTS
    return total


def decide(
    candidate: ModelResult,
    baseline: ModelResult,
    all_candidates: list[ModelResult],
) -> Verdict:
    """Apply the full decision rule.

    `all_candidates` includes the baseline. The function picks the top-scoring
    non-vetoed candidate and applies the non-decision branch against the baseline.
    """
    scores: list[CandidateScore] = []
    for r in all_candidates:
        vetoes = apply_vetoes(r) if r.model_id != baseline.model_id else []
        scores.append(CandidateScore(
            model_id=r.model_id,
            total=weighted_total(r),
            veto_reasons=vetoes,
            eligible=not vetoes,
        ))

    scores.sort(key=lambda s: s.total, reverse=True)
    baseline_score = next(s for s in scores if s.model_id == baseline.model_id)

    # Filter to eligible non-baseline candidates
    eligible_candidates = [s for s in scores if s.eligible and s.model_id != baseline.model_id]

    if not eligible_candidates:
        return Verdict(
            verdict="stay",
            winner_id=baseline.model_id,
            scores=scores,
            veto_reasons=[r for s in scores for r in s.veto_reasons],
            rationale="All non-baseline candidates were vetoed on hard floors.",
        )

    top_candidate = eligible_candidates[0]

    # Non-decision: top candidate within 3pts of baseline AND
    # every source_type within 5pts → stay
    top_result = next(r for r in all_candidates if r.model_id == top_candidate.model_id)
    within_total = abs(top_candidate.total - baseline_score.total) <= NON_DECISION_TOTAL_TOL
    within_source = all(
        abs(top_result.e2e_recall_by_source_type.get(st, 0)
            - top_result.baseline_e2e_recall_by_source_type.get(st, 0))
        <= NON_DECISION_SOURCE_TOL
        for st in top_result.e2e_recall_by_source_type
    )
    if within_total and within_source:
        return Verdict(
            verdict="stay",
            winner_id=baseline.model_id,
            scores=scores,
            veto_reasons=[],
            rationale=f"Top candidate {top_candidate.model_id} within {NON_DECISION_TOTAL_TOL}pts "
                      f"of baseline AND no source_type regresses >{NON_DECISION_SOURCE_TOL}pts. "
                      f"Non-decision branch → stay on Gemini.",
        )

    # Switch verdict (note: actual switch precondition is the 10K-chunk re-embed
    # parity check — that's done outside this function as a post-decision gate.)
    return Verdict(
        verdict="switch",
        winner_id=top_candidate.model_id,
        scores=scores,
        veto_reasons=[],
        rationale=f"Top candidate {top_candidate.model_id} ({top_candidate.total:.1f}) "
                  f"beats baseline ({baseline_score.total:.1f}) by "
                  f"{top_candidate.total - baseline_score.total:.1f}pts. "
                  f"Switch verdict — gated on 10K-chunk parity check.",
    )
