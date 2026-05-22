"""Apply spec §5.4 decision rule to the Phase A artifacts and write a
1-page decision Markdown.

Reuses compute_phase_a_verdict() from build_report.py to keep the rule
single-sourced.
"""
from __future__ import annotations
import argparse
import datetime as dt
import glob
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BENCH_ROOT = Path(__file__).resolve().parents[2]
RAW = BENCH_ROOT / "reports" / "raw"
DEFAULT_OUT = BENCH_ROOT / "reports" / f"{dt.date.today().isoformat()}_phase_a_decision.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    diff_files = sorted(glob.glob(str(RAW / "*_phase_a_static_diff.json")))
    judge_files = sorted(glob.glob(str(RAW / "*_phase_a_judge.json")))
    if not diff_files or not judge_files:
        log.error("Missing phase_a_static_diff.json or phase_a_judge.json. Run runners 4 + 6 first.")
        sys.exit(1)

    static_diff = json.loads(Path(diff_files[-1]).read_text())
    judge = json.loads(Path(judge_files[-1]).read_text())

    sys.path.insert(0, str(BENCH_ROOT / "runners"))
    from build_report import compute_phase_a_verdict
    verdict = compute_phase_a_verdict(static_diff, judge)

    # Pull snapshot identity into the decision doc
    sys.path.insert(0, str(BENCH_ROOT / "src"))
    from bench.snapshot import SNAPSHOT_ID, CASE_KB_COMMIT

    lines = [
        f"# Phase A Decision — {dt.date.today().isoformat()}",
        "",
        f"**Verdict:** `{verdict['verdict']}` &middot; **{verdict['label']}**",
        "",
        f"**Rationale:** {verdict['rationale']}",
        "",
        f"**Snapshot:** `{SNAPSHOT_ID}` (case_kb commit `{CASE_KB_COMMIT}`)",
        "",
        "## Headline numbers",
        "",
        f"- Median `jaccard@10`: **{verdict.get('median_jaccard', 0):.3f}**",
        f"- Median `cited_weighted_overlap` (Harrier): **{verdict.get('median_cited_weighted', 0):.3f}**",
        f"- Sonnet judge — sufficient/better_than_gemini: **{verdict.get('n_sufficient_or_better', 0)}** of **{verdict.get('n_total_turns', 0)}** turns",
        f"- Sonnet judge — insufficient: **{verdict.get('n_insufficient', 0)}** turns",
        "",
        "## Decision rule (spec §5.4, locked before run)",
        "",
        "- **A1 — competitive:** median jaccard@10 ≥ 0.90 AND median cited_weighted ≥ 0.95 AND sufficient/better ≥ 9 of 11 turns",
        "- **A2 — clearly loses:** median jaccard@10 < 0.70 OR median cited_weighted < 0.80 OR insufficient ≥ 5 of 11 turns",
        "- **A3 — inconclusive:** anything in between",
        "",
        "## Next action",
        "",
    ]
    if verdict["verdict"] == "A1":
        lines.append("Harrier is competitive at retrieval-level on the snapshot. **Proceed to Phase B** — full multi-turn pi-mono replay (separate spec) with N=3 replicates, two judges, fact-grounded correctness, privilege-leak axis.")
    elif verdict["verdict"] == "A2":
        lines.append("Harrier clearly loses on the actual queries the agent makes. **Stay on Gemini.** Phase B is not worth running until a different candidate (e.g., FP8 Qwen3 via vLLM, AWQ INT4) is added to the bench.")
    else:
        lines.append("**User decision required.** Options: (a) run Phase B for a clearer answer (~$15-30, 14-20h dev), (b) defer to post-trial, (c) accept current Gemini setup.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    log.info(f"Wrote {args.out}")
    log.info(f"Verdict: {verdict['verdict']} — {verdict['label']}")


if __name__ == "__main__":
    main()
