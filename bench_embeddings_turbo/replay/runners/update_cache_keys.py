"""Compute SHA-keyed cache invalidation values per spec §3.3.

Snapshot-aligned: hashes the snapshot files (frozen) and the bench code that
operates on them. Two literal-constant keys pin the snapshot identity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bench.snapshot import (
    SNAPSHOT_ID, CASE_KB_COMMIT,
    SNAPSHOT_LANCEDB_PATH, SNAPSHOT_CHUNKS_PARQUET,
    SNAPSHOT_SEARCH_DUCKDB, SNAPSHOT_FACTS_JSONL,
    assert_snapshot_present,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BENCH_ROOT = Path(__file__).resolve().parents[2]
CACHE_KEYS_PATH = BENCH_ROOT / "cache_keys.json"


def _sha_of_file(p: Path) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _sha_of_dir(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    for child in sorted(p.rglob("*")):
        if child.is_file():
            h.update(child.relative_to(p).as_posix().encode())
            h.update(b"\x00")
            h.update(child.read_bytes())
    return h.hexdigest()


def compute_cache_keys() -> dict[str, str | None]:
    return {
        # Snapshot identity — pinned constants
        "snapshot_id":                  SNAPSHOT_ID,
        "case_kb_commit":               CASE_KB_COMMIT,

        # Snapshot file content SHAs — invalidate cache if any snapshot file changes
        "snapshot_chunks_parquet_sha":  _sha_of_file(SNAPSHOT_CHUNKS_PARQUET),
        "snapshot_search_duckdb_sha":   _sha_of_file(SNAPSHOT_SEARCH_DUCKDB),
        "snapshot_facts_jsonl_sha":     _sha_of_file(SNAPSHOT_FACTS_JSONL),
        "snapshot_lancedb_dir_sha":     _sha_of_dir(SNAPSHOT_LANCEDB_PATH),

        # Bench code SHAs — invalidate if eval logic / schema / scoring changes
        "bench_snapshot_py_sha":        _sha_of_file(BENCH_ROOT / "src" / "bench" / "snapshot.py"),
        "bench_schemas_py_sha":         _sha_of_file(BENCH_ROOT / "src" / "bench" / "schemas.py"),
        "bench_scoring_py_sha":         _sha_of_file(BENCH_ROOT / "src" / "bench" / "scoring.py"),
        "bench_source_weights_py_sha":  _sha_of_file(BENCH_ROOT / "src" / "bench" / "source_weights.py"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CACHE_KEYS_PATH)
    args = ap.parse_args()

    assert_snapshot_present()
    keys = compute_cache_keys()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(keys, indent=2))
    log.info(f"Wrote {args.out}")
    for k, v in keys.items():
        short = (v[:16] + "...") if v and len(v) > 20 else v
        log.info(f"  {k}: {short}")


if __name__ == "__main__":
    main()
