"""Extract the final structured JSON block from each Phase B subagent transcript.

Reads only the LAST assistant message of each JSONL transcript via streaming,
parses out the fenced ```json block, and writes one parsed JSON per subagent
under reports/raw/phase_b/.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


SUBAGENTS = {
    "pb_v1_r1": "a0ada5447787826af",
    "pb_v1_r2": "a55ea4eafdd644b4e",
    "pb_v1_r3": "aa29e73c597d4b854",
    "pb_v2_r1": "a9fa07a83e3f6de8a",
    "pb_v2_r2": "a0f11990d11b8b4db",
    "pb_v2_r3": "a58035f6f4a47a250",
    "pb_qwen_r1": "a055654a06a586cfb",
    "pb_qwen_r2": "a14c9ba649edeb73e",
    "pb_qwen_r3": "abf749d4682a73b9d",
}

TASKS_DIR = Path("/tmp/claude-1000/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/tasks")
OUT = Path("/mnt/i/dev/Legal/case_kb/bench_embeddings/reports/raw/phase_b")
OUT.mkdir(parents=True, exist_ok=True)

# Regex to grab the LAST fenced ```json block in the text
JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def last_assistant_text(jsonl_path: Path) -> str:
    """Stream the JSONL and return the text of the LAST assistant message."""
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
    """Pull the LAST fenced JSON block from text and parse it."""
    matches = JSON_FENCE.findall(text)
    if not matches:
        return None
    # Try parsing from the LAST match backwards (the structured output is at end)
    for block in reversed(matches):
        try:
            return json.loads(block)
        except Exception:
            continue
    return None


def main() -> None:
    summary = {}
    for name, task_id in SUBAGENTS.items():
        # Some platforms wrap the file in subagents/ instead of tasks/
        candidates = [
            TASKS_DIR / f"{task_id}.output",
            Path(f"/home/tommy/.claude/projects/-mnt-i-dev-Legal/57bff620-6070-4181-8cb9-99db8ce90f5b/subagents/agent-{task_id}.jsonl"),
        ]
        src = next((p for p in candidates if p.exists() and p.stat().st_size > 200), None)
        if src is None:
            print(f"[{name}] NOT FOUND in any candidate path", file=sys.stderr)
            continue
        text = last_assistant_text(src)
        parsed = extract_json_block(text)
        if parsed is None:
            print(f"[{name}] last assistant had no parseable JSON block (len={len(text)})", file=sys.stderr)
            continue
        out_path = OUT / f"{name}.json"
        out_path.write_text(json.dumps(parsed, indent=2))
        n_turns = len(parsed.get("turns", []))
        n_searches = sum(len(t.get("searches", [])) for t in parsed.get("turns", []))
        backend = parsed.get("backend", "?")
        rep = parsed.get("replicate", "?")
        summary[name] = {
            "backend": backend,
            "replicate": rep,
            "n_turns": n_turns,
            "n_searches": n_searches,
            "size": out_path.stat().st_size,
        }
        print(f"[{name}] backend={backend} rep={rep} turns={n_turns} searches={n_searches} -> {out_path}")

    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"Wrote {len(summary)}/9 subagent JSONs to {OUT}")


if __name__ == "__main__":
    main()
