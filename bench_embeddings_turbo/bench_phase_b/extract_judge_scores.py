"""Extract and consolidate judge scores from the 15 judge subagent transcripts."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from statistics import mean


JUDGES = {
    # query_id -> {judge_label -> agent_id}
    "parental_alienation": {
        "sonnet": "a831ac7ae47f6d227",
        "opus": "aa93c7e59ddd174fb",
        "skeptical": "a0af77ddc39a8f840",
    },
    "access_denial_synonyms": {
        "sonnet": "a153ab19b37ef9779",
        "opus": "adda5f68474ef2a06",
        "skeptical": "a2b8354fcd4b510ae",
    },
    "valley_point_event": {
        "sonnet": "aef5256f42b66c129",
        "opus": "abc37cbefd33c8971",
        "skeptical": "a3a0b74ee3a26ff28",
    },
    "disclosure_non_compliance": {
        "sonnet": "ab4a0f6315e928886",
        "opus": "a06197b67823e1a11",
        "skeptical": "a51e0ac48db705f93",
    },
    "indirect_contribution_caregiver": {
        "sonnet": "a6c88c4451a800395",
        "opus": "ad4d9743bb7f2a504",
        "skeptical": "a1d98cae40b410ac0",
    },
}

TASKS_DIR = Path("/tmp/claude-1000/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/tasks")
SUBAGENT_DIR = Path("/home/tommy/.claude/projects/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/subagents")
OUT = Path("/mnt/i/dev/Legal/case_kb/bench_embeddings/reports/raw/judge_scores")
OUT.mkdir(parents=True, exist_ok=True)

JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


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


def find_transcript(agent_id: str) -> Path | None:
    for p in [TASKS_DIR / f"{agent_id}.output", SUBAGENT_DIR / f"agent-{agent_id}.jsonl"]:
        if p.exists() and p.stat().st_size > 200:
            return p
    return None


def main() -> None:
    consolidated = {}
    for qid, judges in JUDGES.items():
        per_chunk: dict[str, dict] = {}  # chunk_id -> {judge: score}
        for judge_label, agent_id in judges.items():
            tp = find_transcript(agent_id)
            if tp is None:
                print(f"[{qid}/{judge_label}] transcript missing", file=sys.stderr)
                continue
            text = last_assistant_text(tp)
            parsed = extract_json(text)
            if parsed is None:
                print(f"[{qid}/{judge_label}] no parseable JSON", file=sys.stderr)
                continue
            for s in parsed.get("scores", []):
                cid = s.get("chunk_id")
                score = s.get("score")
                if cid is None or score is None:
                    continue
                per_chunk.setdefault(cid, {})[judge_label] = int(score)

        # Compute consensus per chunk
        for cid, scores in per_chunk.items():
            if scores:
                vals = list(scores.values())
                per_chunk[cid]["mean"] = round(mean(vals), 2)
                per_chunk[cid]["max"] = max(vals)
                per_chunk[cid]["min"] = min(vals)
                per_chunk[cid]["variance"] = max(vals) - min(vals)
                per_chunk[cid]["n_judges"] = len(vals)
                # Consensus relevant if mean >= 1.5 (i.e. typically scored >= 2 by majority)
                per_chunk[cid]["consensus_relevant"] = per_chunk[cid]["mean"] >= 1.5

        consolidated[qid] = per_chunk
        n_total = len(per_chunk)
        n_rel = sum(1 for c in per_chunk.values() if c.get("consensus_relevant"))
        n_disagree = sum(1 for c in per_chunk.values() if c.get("variance", 0) >= 2)
        print(f"[{qid}] {n_total} chunks scored, {n_rel} consensus-relevant, {n_disagree} high-variance disagreements")

    out_path = OUT / "consolidated_scores.json"
    out_path.write_text(json.dumps(consolidated, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
