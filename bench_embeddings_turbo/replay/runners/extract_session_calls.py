"""Extract every search_evidence + build_evidence_pack call from the pi-session
HTML, along with per-turn user text and cited evidence_ids.

Output: bench_embeddings/replay/data/replay/session_calls.json

Pure file-parsing — does not touch production data, so no snapshot assertion
needed (Task 1 already validated the snapshot).
"""
from __future__ import annotations
import argparse
import base64
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

INLINE_ID_RE = re.compile(r"\[([a-zA-Z0-9._@\$\-]{12,})\]")
DEFAULT_HTML = Path("/mnt/i/dev/Legal/case_kb/agent/pi-session-2026-04-01T05-26-01-675Z_ccf94dfb-f6d2-48f3-b40f-ef534a97268a.html")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "replay" / "session_calls.json"


def _load_session(html_path: Path) -> dict:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'<script id="session-data" type="application/json">([^<]+)</script>', text)
    if not m:
        raise ValueError(f"No session-data block in {html_path}")
    return json.loads(base64.b64decode(m.group(1).strip()))


def _ids_from_tool_result(content: Any) -> list[str]:
    blob = content if isinstance(content, str) else json.dumps(content)
    ids = list(INLINE_ID_RE.findall(blob))
    for m in re.finditer(r"Evidence ID:\s*([a-zA-Z0-9._@\$\-]{12,})", blob):
        ids.append(m.group(1))
    seen, out = set(), []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_session_calls(html_path: Path, out_path: Path) -> list[dict]:
    data = _load_session(html_path)
    entries = data.get("entries", [])

    out: list[dict] = []
    turn_idx = 0
    current_user_text = None
    for e in entries:
        if e.get("type") != "message":
            continue
        msg = e["message"]
        role = msg.get("role")
        content = msg.get("content", []) or []

        if role == "user":
            turn_idx += 1
            current_user_text = ""
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    current_user_text += c.get("text", "")

        if role == "assistant":
            for c in content:
                if not isinstance(c, dict) or c.get("type") != "toolCall":
                    continue
                if c.get("name") not in ("search_evidence", "build_evidence_pack"):
                    continue
                out.append({
                    "turn_idx": turn_idx,
                    "user_text": current_user_text,
                    "tool_call_id": c.get("id"),
                    "tool_name": c.get("name"),
                    "args": c.get("arguments") or {},
                    "returned_chunk_ids": [],
                })

        if role == "toolResult":
            tcid = msg.get("toolCallId") or msg.get("tool_call_id") or msg.get("id")
            for record in out:
                if record["tool_call_id"] == tcid:
                    record["returned_chunk_ids"] = _ids_from_tool_result(msg.get("content"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def extract_cited_per_turn(html_path: Path) -> dict[int, list[str]]:
    """Per turn: every chunk_id the agent cited (save_case_fact, read_document, inline text)."""
    data = _load_session(html_path)
    entries = data.get("entries", [])
    cited: dict[int, list[str]] = {}
    turn_idx = 0
    for e in entries:
        if e.get("type") != "message":
            continue
        msg = e["message"]
        role = msg.get("role")
        if role == "user":
            turn_idx += 1
            cited.setdefault(turn_idx, [])
            continue
        if role != "assistant":
            continue
        for c in msg.get("content", []) or []:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                cited.setdefault(turn_idx, []).extend(INLINE_ID_RE.findall(c.get("text", "")))
            elif c.get("type") == "toolCall":
                args = c.get("arguments") or {}
                if c.get("name") == "save_case_fact":
                    cited.setdefault(turn_idx, []).extend(args.get("evidence_ids", []) or [])
                elif c.get("name") == "read_document":
                    eid = args.get("evidence_id")
                    if eid:
                        cited.setdefault(turn_idx, []).append(eid)
    return {t: sorted(set(v)) for t, v in cited.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    log.info(f"Parsing {args.html}...")
    calls = extract_session_calls(args.html, args.out)
    log.info(f"  Extracted {len(calls)} tool calls")
    cited = extract_cited_per_turn(args.html)
    log.info(f"  {sum(len(v) for v in cited.values())} cited evidence_ids across {len(cited)} turns")

    payload = {"calls": calls, "cited_per_turn": {str(k): v for k, v in cited.items()}}
    args.out.write_text(json.dumps(payload, indent=2))
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
