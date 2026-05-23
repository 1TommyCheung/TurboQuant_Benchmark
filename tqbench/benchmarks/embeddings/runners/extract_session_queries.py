"""Extract real search queries from pi-agent sessions.

Reads:
- agent/pi-session-*.html  (base64 JSON in <script id="session-data">)
- data/evidence_lake/derived/sessions/*.json

Writes:
- data/eval_queries/_layer1_raw.jsonl  (one record per query)

Each record:
{
  "id": "session_<id>_call_<n>",
  "query": "<the raw query string>",
  "filters": {...},               # source_type, party, date_from, date_to, top_k
  "tool": "search_evidence" | "build_evidence_pack",
  "cited_evidence_ids": [...],    # extracted from save_case_fact results in the same session
  "source_file": "<absolute path>"
}
"""
from __future__ import annotations
import argparse
import base64
import json
import logging
import re
from pathlib import Path

CASE_KB_ROOT = Path(__file__).resolve().parents[4]
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_queries" / "_layer1_raw.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _extract_from_html(html_path: Path) -> list[dict]:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'<script id="session-data" type="application/json">([^<]+)</script>', text)
    if not m:
        log.warning(f"No session-data script in {html_path}")
        return []
    data = json.loads(base64.b64decode(m.group(1).strip()))
    cited: set[str] = set()

    # First pass: collect cited evidence_ids from save_case_fact tool calls
    for e in data.get("entries", []):
        if e.get("type") != "message":
            continue
        msg = e["message"]
        if msg.get("role") != "assistant":
            continue
        for c in msg.get("content", []):
            if isinstance(c, dict) and c.get("type") == "toolCall" and c.get("name") == "save_case_fact":
                args = c.get("arguments") or {}
                cited.update(args.get("evidence_ids", []) or [])

    # Second pass: extract queries
    rows: list[dict] = []
    for e in data.get("entries", []):
        if e.get("type") != "message":
            continue
        msg = e["message"]
        if msg.get("role") != "assistant":
            continue
        for c in msg.get("content", []):
            if not isinstance(c, dict) or c.get("type") != "toolCall":
                continue
            name = c.get("name")
            args = c.get("arguments") or {}
            if name == "search_evidence":
                q = args.get("query")
                if not q:
                    continue
                filters = {k: v for k, v in args.items() if k != "query"}
                rows.append({
                    "id": f"{data.get('header', {}).get('id', html_path.stem)}_{c.get('id')}",
                    "query": q,
                    "filters": filters,
                    "tool": "search_evidence",
                    "cited_evidence_ids": sorted(cited),
                    "source_file": str(html_path),
                })
            elif name == "build_evidence_pack":
                claim = args.get("claim")
                if not claim:
                    continue
                filters = {k: v for k, v in args.items() if k != "claim"}
                rows.append({
                    "id": f"{data.get('header', {}).get('id', html_path.stem)}_{c.get('id')}",
                    "query": claim,
                    "filters": filters,
                    "tool": "build_evidence_pack",
                    "cited_evidence_ids": sorted(cited),
                    "source_file": str(html_path),
                })
    return rows


def _extract_from_json(json_path: Path) -> list[dict]:
    """Parse the older 3-session JSON format (turns with query + chunk_ids)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for i, turn in enumerate(data.get("turns", [])):
        q = turn.get("query")
        if not q:
            continue
        rows.append({
            "id": f"{data.get('session_id', json_path.stem)}_turn_{i}",
            "query": q,
            "filters": {},
            "tool": "rag_chat",
            "cited_evidence_ids": sorted(turn.get("chunk_ids", []) or []),
            "source_file": str(json_path),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-glob", default=str(CASE_KB_ROOT / "agent" / "pi-session-*.html"))
    ap.add_argument("--json-glob", default=str(
        CASE_KB_ROOT.parent / "data" / "evidence_lake" / "derived" / "sessions" / "*.json"
    ))
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    import glob
    all_rows: list[dict] = []
    for p in sorted(glob.glob(args.html_glob)):
        rows = _extract_from_html(Path(p))
        log.info(f"  {p}: {len(rows)} queries")
        all_rows.extend(rows)
    for p in sorted(glob.glob(args.json_glob)):
        rows = _extract_from_json(Path(p))
        log.info(f"  {p}: {len(rows)} queries")
        all_rows.extend(rows)

    # Deduplicate by query text, keep first occurrence
    seen: set[str] = set()
    unique: list[dict] = []
    for r in all_rows:
        if r["query"].strip().lower() in seen:
            continue
        seen.add(r["query"].strip().lower())
        unique.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in unique:
            f.write(json.dumps(r) + "\n")
    log.info(f"Wrote {len(unique)} unique queries to {args.out}")


if __name__ == "__main__":
    main()
