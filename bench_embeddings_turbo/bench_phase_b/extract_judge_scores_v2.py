"""Auto-discover judge transcripts (15 queries × 3 judges) and consolidate scores.

Scans every agent-*.jsonl transcript in the subagents directory. For each one,
looks at the first user message to determine:
  - judge_label (sonnet / opus / skeptical) via intro signature
  - query_id by matching the "Query:" line against the known query list

Then extracts the last ```json block from the assistant's reply, validates the
embedded query_id matches what we inferred, and consolidates per-chunk scores.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from statistics import mean

SUBAGENT_DIR = Path("/home/tommy/.claude/projects/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/subagents")
TASKS_DIR = Path("/tmp/claude-1000/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/tasks")
OUT = Path("/mnt/i/dev/Legal/case_kb/bench_embeddings/reports/raw/judge_scores")
OUT.mkdir(parents=True, exist_ok=True)

# All 15 queries — id -> query string (used to fingerprint prompts)
QUERIES = {
    "parental_alienation": "parental alienation children refusing access negative perception father",
    "access_denial_synonyms": "access denied refused blocked cancelled children",
    "valley_point_event": "Valley Point Shopping Centre encounter Tristan November 2025",
    "disclosure_non_compliance": "disclosure non-compliance failure produce documents adverse inference",
    "indirect_contribution_caregiver": "indirect contribution caregiver homemaker non-financial role sacrifice career",
    "valley_point_handover": "Valley Point pick up drop off children",
    "negative_characterisation": "negative characterisation father children alienation influence",
    "children_refuse_visit": "children don't want to see father refuse visit access",
    "tracy_limited_access_proposals": "limited meal only access proposals suspended children",
    "hk_unilateral_travel": "Hong Kong children taken without consent passports missing",
    "father_school_visit_framing": "father visit school kindergarten approaching children negative framing harassment",
    "counselling_engagement_pattern": "counselling sessions in-person tele-conference engagement willingness refusal",
    "gatekeeping_pattern": "gatekeeping behaviour mother exclude father co-parenting interference",
    "fdr_mediation_directions": "FDR mediation directions counsellor third party suspension access",
    "matrimonial_home_valuation": "matrimonial home valuation arms-length sale market value division",
}

JUDGE_SIGNATURES = {
    "careful neutral evidence-relevance judge": "sonnet",
    "senior legal researcher (Opus)": "opus",
    "adversarial reviewer (Skeptical Opus)": "skeptical",
}

JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
PROMPT_FNAME_RE = re.compile(r"judge_pools/([a-z_]+)__(sonnet|opus|skeptical_opus)\.prompt\.txt")


def first_user_text(jsonl_path: Path) -> str:
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "user":
                continue
            msg = rec.get("message", {})
            content = msg.get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "text":
                        return c.get("text", "")
            elif isinstance(content, str):
                return content
    return ""


def last_assistant_text(jsonl_path: Path) -> str:
    last = ""
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "assistant":
                continue
            msg = rec.get("message", {})
            content = msg.get("content")
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                last = "\n".join(texts)
            elif isinstance(content, str):
                last = content
    return last


def extract_json(text: str) -> dict | None:
    matches = JSON_FENCE.findall(text)
    if not matches:
        return None
    for block in reversed(matches):
        try:
            return json.loads(block)
        except Exception:
            pass
        try:
            return json.loads(block, strict=False)
        except Exception:
            continue
    return None


def identify_prompt(prompt_text: str) -> tuple[str | None, str | None]:
    """Return (query_id, judge_label) by parsing the filename in the user message."""
    m = PROMPT_FNAME_RE.search(prompt_text)
    if not m:
        return None, None
    qid = m.group(1)
    judge_raw = m.group(2)
    judge = "skeptical" if judge_raw == "skeptical_opus" else judge_raw
    if qid not in QUERIES:
        return None, None
    return qid, judge


def alt_task_text(agent_id: str) -> str:
    p = TASKS_DIR / f"{agent_id}.output"
    if p.exists():
        return p.read_text(errors="ignore")
    return ""


def main() -> None:
    # Map agent_id -> (qid, judge_label)
    discovered: dict[tuple[str, str], list[Path]] = {}
    for tp in sorted(SUBAGENT_DIR.glob("agent-*.jsonl")):
        if tp.stat().st_size < 200:
            continue
        prompt = first_user_text(tp)
        qid, judge = identify_prompt(prompt)
        if qid is None or judge is None:
            continue
        discovered.setdefault((qid, judge), []).append(tp)

    print(f"Discovered {len(discovered)} (query, judge) combos across {sum(len(v) for v in discovered.values())} transcripts")

    expected = len(QUERIES) * 3
    if len(discovered) != expected:
        missing = []
        for qid in QUERIES:
            for j in ["sonnet", "opus", "skeptical"]:
                if (qid, j) not in discovered:
                    missing.append(f"{qid}/{j}")
        print(f"WARNING: expected {expected} combos, missing: {missing}")

    consolidated: dict[str, dict] = {}
    for (qid, judge), tps in discovered.items():
        # Use most recent transcript (largest mtime) if multiples
        tp = max(tps, key=lambda p: p.stat().st_mtime)
        text = last_assistant_text(tp)
        parsed = extract_json(text)
        if parsed is None:
            # Try the .output file
            agent_id = tp.stem.replace("agent-", "")
            alt = alt_task_text(agent_id)
            parsed = extract_json(alt)
        if parsed is None:
            print(f"[{qid}/{judge}] NO JSON parsed (agent={tp.name})", file=sys.stderr)
            continue

        per_chunk = consolidated.setdefault(qid, {})
        for s in parsed.get("scores", []):
            cid = s.get("chunk_id")
            score = s.get("score")
            if cid is None or score is None:
                continue
            try:
                score_int = int(score)
            except (TypeError, ValueError):
                continue
            per_chunk.setdefault(cid, {})[judge] = score_int

    # Compute consensus per chunk per query
    for qid, per_chunk in consolidated.items():
        for cid, scores in per_chunk.items():
            # filter only judge-score keys
            judge_vals = [v for k, v in scores.items() if k in ("sonnet", "opus", "skeptical")]
            if judge_vals:
                scores["mean"] = round(mean(judge_vals), 2)
                scores["max"] = max(judge_vals)
                scores["min"] = min(judge_vals)
                scores["variance"] = max(judge_vals) - min(judge_vals)
                scores["n_judges"] = len(judge_vals)
                scores["consensus_relevant"] = scores["mean"] >= 1.5
        n_total = len(per_chunk)
        n_rel = sum(1 for c in per_chunk.values() if c.get("consensus_relevant"))
        n_strict = sum(1 for c in per_chunk.values() if c.get("mean", 0) >= 2.5)
        n_disagree = sum(1 for c in per_chunk.values() if c.get("variance", 0) >= 2)
        print(f"[{qid}] {n_total} chunks, {n_rel} consensus≥1.5, {n_strict} strict≥2.5, {n_disagree} high-variance")

    out_path = OUT / "consolidated_scores.json"
    out_path.write_text(json.dumps(consolidated, indent=2))
    print(f"\nWrote {out_path}  ({len(consolidated)} queries)")


if __name__ == "__main__":
    main()
