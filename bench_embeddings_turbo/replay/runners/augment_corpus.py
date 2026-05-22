"""Augment bench LanceDB tables with chunk_ids referenced by any pi-session.

Reads exclusively from the FROZEN snapshot (bench.snapshot) — production
LanceDB is never touched directly. See SNAPSHOT.md.

For every chunk_id appearing in a search_evidence/build_evidence_pack tool
RESULT, a read_document call's evidence_id, a save_case_fact call's
evidence_ids list, or inline [uuid] text in the assistant's output:

  - Look it up in the snapshot LanceDB.
  - For Gemini bench LanceDB: copy the vector directly (zero API call).
  - For Harrier bench LanceDB: encode the chunk_text with Harrier and append.
"""
from __future__ import annotations
import argparse
import base64
import json
import logging
import re
import sys
from pathlib import Path

import lancedb

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bench.io_lance import BENCH_LANCEDB_ROOT
from bench.models import load_registry, load_embedder
from bench.snapshot import SNAPSHOT_LANCEDB_PATH, assert_snapshot_present

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

INLINE_ID_RE = re.compile(r"\[([a-zA-Z0-9._@\$\-]{12,})\]")


def _extract_evidence_ids_from_save_case_fact_args(args: dict) -> set[str]:
    raw = args.get("evidence_ids") or []
    if not isinstance(raw, list):
        return set()
    return {str(x) for x in raw if x}


def _extract_inline_evidence_ids(text: str) -> set[str]:
    if not text:
        return set()
    return set(INLINE_ID_RE.findall(text))


def extract_referenced_chunk_ids(html_path: Path) -> set[str]:
    """Walk the pi-session HTML's embedded JSON entries and collect every
    chunk_id / evidence_id mentioned in tool calls, tool results, or
    assistant text."""
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'<script id="session-data" type="application/json">([^<]+)</script>', text)
    if not m:
        log.warning(f"No session-data block in {html_path}")
        return set()
    data = json.loads(base64.b64decode(m.group(1).strip()))

    ids: set[str] = set()
    for e in data.get("entries", []):
        if e.get("type") != "message":
            continue
        msg = e["message"]
        role = msg.get("role")
        content = msg.get("content", []) or []

        if role == "assistant":
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    ids |= _extract_inline_evidence_ids(c.get("text", ""))
                elif c.get("type") == "toolCall":
                    name = c.get("name", "")
                    args = c.get("arguments") or {}
                    if name == "save_case_fact":
                        ids |= _extract_evidence_ids_from_save_case_fact_args(args)
                    elif name == "read_document":
                        eid = args.get("evidence_id")
                        if eid:
                            ids.add(str(eid))

        if role == "toolResult":
            raw = msg.get("content")
            blob = raw if isinstance(raw, str) else json.dumps(raw)
            ids |= _extract_inline_evidence_ids(blob)
            for m2 in re.finditer(r"Evidence ID:\s*([a-zA-Z0-9._@\$\-]{12,})", blob):
                ids.add(m2.group(1))

    return ids


def compute_augment_set(
    referenced_chunk_ids: set[str],
    bench_chunk_ids: set[str],
) -> set[str]:
    """Return referenced chunk_ids not yet in the bench table.

    Caller must pre-filter `referenced_chunk_ids` to the snapshot's chunk_id
    universe — use expand_evidence_ids_to_chunk_ids() for that.
    """
    return referenced_chunk_ids - bench_chunk_ids


def expand_evidence_ids_to_chunk_ids(evidence_ids: set[str]) -> set[str]:
    """For each evidence_id, return all chunk_ids that belong to it in the snapshot.

    The pi-session HTML references parent documents by evidence_id (save_case_fact,
    read_document, inline [uuid] in assistant text). Each evidence document has 1..N
    retrievable chunks in the snapshot — augmentation works in chunk_id space, so we
    expand here.
    """
    if not evidence_ids:
        return set()
    db = lancedb.connect(str(SNAPSHOT_LANCEDB_PATH))
    tbl = db.open_table("chunks")
    n = tbl.count_rows()
    df = tbl.search().select(["chunk_id", "evidence_id"]).limit(n).to_pandas()
    return set(df[df["evidence_id"].isin(evidence_ids)]["chunk_id"].astype(str))


def _read_chunk_ids(table) -> set[str]:
    df = table.to_pandas()
    return set(df["chunk_id"].astype(str))


def _length_bucket(token_count) -> str:
    tc = int(token_count or 0)
    if tc < 512: return "short"
    if tc < 2048: return "medium"
    if tc < 8192: return "long"
    return "very_long"


def augment_gemini_table(augment_set: set[str], spec) -> int:
    """For Gemini, copy snapshot vectors directly into the bench table."""
    db_bench = lancedb.connect(str(BENCH_LANCEDB_ROOT))
    bench_tbl = db_bench.open_table(spec.id)

    db_snap = lancedb.connect(str(SNAPSHOT_LANCEDB_PATH))
    snap_tbl = db_snap.open_table("chunks")
    snap_df = snap_tbl.to_pandas()
    snap_df = snap_df[snap_df["chunk_id"].isin(augment_set)]
    if snap_df.empty:
        return 0

    rows = []
    for _, row in snap_df.iterrows():
        rows.append({
            "chunk_id": row["chunk_id"],
            "evidence_id": row.get("evidence_id", ""),
            "source_type": row.get("source_type", ""),
            "length_bucket": _length_bucket(row.get("token_count", 0)),
            "token_count": int(row.get("token_count", 0) or 0),
            "date_sgt": str(row.get("date_sgt", "")),
            "party_from": str(row.get("party_from", "")),
            "vector": row["vector"].tolist() if hasattr(row["vector"], "tolist") else list(row["vector"]),
        })
    if rows:
        bench_tbl.add(rows)
    return len(rows)


def augment_local_table(augment_set: set[str], spec) -> int:
    """For local candidates, encode the chunk_text with that model."""
    db_bench = lancedb.connect(str(BENCH_LANCEDB_ROOT))
    bench_tbl = db_bench.open_table(spec.id)

    db_snap = lancedb.connect(str(SNAPSHOT_LANCEDB_PATH))
    snap_tbl = db_snap.open_table("chunks")
    snap_df = snap_tbl.to_pandas()
    snap_df = snap_df[snap_df["chunk_id"].isin(augment_set)]
    if snap_df.empty:
        return 0

    embedder = load_embedder(spec.id)
    texts = snap_df["chunk_text"].tolist()
    vecs = embedder.encode(texts, batch_size=4)

    rows = []
    for (_, row), v in zip(snap_df.iterrows(), vecs):
        rows.append({
            "chunk_id": row["chunk_id"],
            "evidence_id": row.get("evidence_id", ""),
            "source_type": row.get("source_type", ""),
            "length_bucket": _length_bucket(row.get("token_count", 0)),
            "token_count": int(row.get("token_count", 0) or 0),
            "date_sgt": str(row.get("date_sgt", "")),
            "party_from": str(row.get("party_from", "")),
            "vector": v.astype("float32").tolist(),
        })
    if rows:
        bench_tbl.add(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-html", type=Path,
                    default=Path("/mnt/i/dev/Legal/case_kb/agent/pi-session-2026-04-01T05-26-01-675Z_ccf94dfb-f6d2-48f3-b40f-ef534a97268a.html"))
    args = ap.parse_args()

    assert_snapshot_present()

    log.info("Extracting referenced ids from pi-session HTML...")
    referenced = extract_referenced_chunk_ids(args.session_html)
    log.info(f"  {len(referenced)} unique referenced ids")

    log.info("Expanding referenced evidence_ids to chunk_ids...")
    referenced_chunk_ids = expand_evidence_ids_to_chunk_ids(referenced)
    log.info(f"  {len(referenced)} referenced evidence_ids -> {len(referenced_chunk_ids)} chunk_ids in snapshot")

    db_bench = lancedb.connect(str(BENCH_LANCEDB_ROOT))
    for spec in load_registry():
        try:
            bench_tbl = db_bench.open_table(spec.id)
        except Exception:
            log.warning(f"  skipping {spec.id}: bench table not present")
            continue
        bench_ids = _read_chunk_ids(bench_tbl)
        log.info(f"{spec.id}: bench has {len(bench_ids):,} chunks")

        augment_set = compute_augment_set(referenced_chunk_ids, bench_ids)
        log.info(f"  augment set: {len(augment_set)} new chunks")

        if not augment_set:
            continue

        if spec.kind == "api":
            n = augment_gemini_table(augment_set, spec)
        else:
            n = augment_local_table(augment_set, spec)
        log.info(f"  added {n} rows to {spec.id} bench table")


if __name__ == "__main__":
    main()
