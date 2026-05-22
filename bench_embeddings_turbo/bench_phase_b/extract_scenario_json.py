"""Extract structured JSON from Phase B+ complex-scenario subagent transcripts."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


SCENARIO_SUBAGENTS = {
    "scenario_1_multi_hop__v1": "af5a71690f39a2f1a",
    "scenario_1_multi_hop__v2": "a9e2a2d1812f20ed9",
    "scenario_1_multi_hop__qwen": "a1b81649e632c863a",
    "scenario_2_timeline__v1": "ad191d25e3629248d",
    "scenario_2_timeline__v2": "ac4527391def5e2ab",
    "scenario_2_timeline__qwen": "a817ef979241b92df",
    "scenario_3_privilege__v1": "a87050a4b96ce01fa",
    "scenario_3_privilege__v2": "a0d1b0511f31c7322",
    "scenario_3_privilege__qwen": "a354bb985c5f5e6b8",
    "scenario_4_correction__v1": "a9278c8450dd7999f",
    "scenario_4_correction__v2": "ae772898620e909ad",
    "scenario_4_correction__qwen": "a05ccadd64dab6550",
}

TASKS_DIR = Path("/tmp/claude-1000/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/tasks")
SUBAGENT_DIR = Path("/home/tommy/.claude/projects/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/subagents")
OUT = Path("/mnt/i/dev/Legal/case_kb/bench_embeddings/reports/raw/phase_b_plus")
OUT.mkdir(parents=True, exist_ok=True)

JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def last_assistant_text(jsonl_path: Path) -> str:
    last_text = ""
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
                last_text = "\n".join(texts)
            elif isinstance(content, str):
                last_text = content
    return last_text


def extract_json_block(text: str) -> dict | None:
    matches = JSON_FENCE.findall(text)
    if not matches:
        return None
    for block in reversed(matches):
        try:
            return json.loads(block)
        except Exception:
            pass
        # Tolerant fallback: allow control characters
        try:
            return json.loads(block, strict=False)
        except Exception:
            continue
    return None


def main() -> None:
    summary = {}
    for name, task_id in SCENARIO_SUBAGENTS.items():
        candidates = [
            TASKS_DIR / f"{task_id}.output",
            SUBAGENT_DIR / f"agent-{task_id}.jsonl",
        ]
        src = next((p for p in candidates if p.exists() and p.stat().st_size > 200), None)
        if src is None:
            print(f"[{name}] NOT FOUND", file=sys.stderr)
            continue
        text = last_assistant_text(src)
        parsed = extract_json_block(text)
        if parsed is None:
            print(f"[{name}] no parseable JSON", file=sys.stderr)
            continue
        out_path = OUT / f"{name}.json"
        out_path.write_text(json.dumps(parsed, indent=2))
        n_turns = len(parsed.get("turns", []))
        n_searches = sum(len(t.get("searches", [])) for t in parsed.get("turns", []))
        unique_chunks = set()
        for t in parsed.get("turns", []):
            for s in t.get("searches", []):
                unique_chunks.update(s.get("top_chunk_ids", []))
            unique_chunks.update(t.get("cited_chunk_ids", []))
        summary[name] = {
            "backend": parsed.get("backend"),
            "scenario": parsed.get("scenario"),
            "skill": parsed.get("skill"),
            "n_turns": n_turns,
            "n_searches": n_searches,
            "unique_chunks_retrieved_or_cited": len(unique_chunks),
        }
        print(f"[{name}] turns={n_turns} searches={n_searches} unique_chunks={len(unique_chunks)}")

    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)}/12 scenario JSONs to {OUT}")


if __name__ == "__main__":
    main()
