"""Snapshot pinning — single source of truth for what this benchmark is calibrated against.

This module hardcodes every production input the benchmark reads. When you
iterate on the case_kb pipeline (re-chunk, re-embed, add sources, change schema),
the bench keeps reading the frozen snapshot defined here, so historical bench
runs remain comparable.

See SNAPSHOT.md in this package root for the full manifest.
"""
from __future__ import annotations
from pathlib import Path

# ============================================================================
# Snapshot identity — DO NOT CHANGE without rebuilding the snapshot data.
# ============================================================================
SNAPSHOT_ID = "2026-05-16_1fe458f"
CASE_KB_COMMIT = "1fe458fa6"  # short hash; full hash recorded in SNAPSHOT.md
SNAPSHOT_DATE = "2026-05-16"
PIPELINE_VERSION = "v4.9"

# ============================================================================
# Snapshot data paths
# ============================================================================
#
# TurboQuant_Benchmark runs as a Windows-native experiment. Source snapshot
# files are read-only; generated LanceDB indexes and reports stay local to
# I:\dev\LLM\TurboQuant_Benchmark\bench_embeddings_turbo.
_DATA_ROOT = Path(r"I:\dev\Legal\data\evidence_lake\snapshots") / f"bench_{SNAPSHOT_ID}"

# Preserved for compatibility with replay scripts. The Windows copy does not
# rely on the old WSL/native-fs snapshot LanceDB; chunk metadata comes from
# SNAPSHOT_CHUNKS_PARQUET in bench.io_lance.read_prod_chunks().
SNAPSHOT_LANCEDB_PATH = _DATA_ROOT / "lancedb_gemini"
SNAPSHOT_CHUNKS_PARQUET = _DATA_ROOT / "chunks.parquet"
SNAPSHOT_SEARCH_DUCKDB = _DATA_ROOT / "search.duckdb"
SNAPSHOT_FACTS_JSONL = _DATA_ROOT / "agent_verified_facts.jsonl"


def assert_snapshot_present() -> None:
    """Fail fast if any expected snapshot file is missing.

    Useful at the top of runners to give a clear error rather than a cryptic
    LanceDB / DuckDB failure deep inside the bench pipeline.
    """
    missing = []
    for label, path in [
        ("chunks.parquet", SNAPSHOT_CHUNKS_PARQUET),
        ("search.duckdb", SNAPSHOT_SEARCH_DUCKDB),
        ("agent_verified_facts.jsonl", SNAPSHOT_FACTS_JSONL),
    ]:
        if not path.exists():
            missing.append(f"  - {label}: {path}")
    if missing:
        raise FileNotFoundError(
            f"Snapshot {SNAPSHOT_ID} is incomplete. Missing:\n"
            + "\n".join(missing)
            + "\n\nSee SNAPSHOT.md for the expected layout."
        )
