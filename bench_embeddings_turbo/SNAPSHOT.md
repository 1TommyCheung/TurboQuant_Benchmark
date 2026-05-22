# Benchmark Snapshot — Frozen Production State

This benchmark is calibrated against a **frozen snapshot** of production data so
that pipeline iteration (re-chunking, re-embedding, new sources, schema changes)
does not invalidate historical bench results.

## Snapshot identity

| Field | Value |
|---|---|
| **Snapshot ID** | `2026-05-16_1fe458f` |
| **Date created** | 2026-05-16 |
| **case_kb commit (anchor)** | `1fe458f7b96daf7f6496ca28d7c066871c58d8df` (`1fe458f`) |
| **Pipeline doc version** | v4.9 |
| **Total size** | ~5.6 GB |

The anchor commit is the last commit that affected production data layout
before the snapshot was taken. Subsequent commits to case_kb (specs, plans,
docs) do not invalidate the snapshot — see "Re-snapshotting" below for when a
new snapshot is required.

## Snapshot contents

Layout mirrors the main codeline's data/index split:

| File | Path | Size | SHA-256 |
|---|---|---|---|
| LanceDB (Gemini, 3072d) | `/home/tommy/evidence_lake_indexes/bench_snapshots/2026-05-16_1fe458f/lancedb_gemini/` | 3.6 GB | (directory — see manifest below) |
| chunks.parquet | `/mnt/i/dev/Legal/data/evidence_lake/snapshots/bench_2026-05-16_1fe458f/chunks.parquet` | 308 MB | `814c570e914850fe3c952368a4ef7e82c09135466db9b9ead68713f508daa75a` |
| search.duckdb (FTS) | `/mnt/i/dev/Legal/data/evidence_lake/snapshots/bench_2026-05-16_1fe458f/search.duckdb` | 1.7 GB | `0b6a5f0bf4470dd186f030d8b8a85090d372dcb1144925546408a6ee7147f31d` |
| agent_verified_facts.jsonl | `/mnt/i/dev/Legal/data/evidence_lake/snapshots/bench_2026-05-16_1fe458f/agent_verified_facts.jsonl` | 8 KB | `2b5c2651f12e280e8a8837babf369dedbea40fdf1d327ab4809b82e84bacb77e` |

A symlink `lancedb_gemini` inside the data snapshot dir points at the native-fs
LanceDB copy, so callers can resolve everything from one root.

### Why LanceDB lives on `/home/tommy/`

Native Linux fs (ext4). LanceDB on NTFS-via-WSL-drvfs breaks Lance file writes
(documented in `case_kb/CLAUDE.md`). The snapshot mirrors this split.

## Production data counts at snapshot time

| Item | Count |
|---|---|
| Emails | 141,177 |
| WhatsApp messages | 88,289 |
| Court docs | 1,144 |
| Solicitor letters | 520 |
| Financial docs | 321 |
| Photos/videos | 5,969 |
| Document exchange records | 6,249 |
| Screenshots | 708 |
| **evidence_master.parquet** | **238,858** |
| **enriched_evidence.parquet** | **110,941** |
| events.parquet | 86,731 |
| **chunks.parquet** | **251,089** |
| **LanceDB vectors (Gemini 3072d)** | **251,089** |
| DuckDB FTS rows | 251,089 |
| agent_verified_facts.jsonl | 11 curated facts |

These numbers should match `case_kb/docs/PIPELINE.md`
§ Data Inventory.

## Schema pin

The production `core/schemas.py:ChunkRecord` definition at commit `1fe458f` is
copied into `src/bench/schemas.py`. The bench uses its own pinned copy so
upstream schema evolution doesn't silently change field interpretation.

If you need to add a new column to `ChunkRecord` upstream, the bench will keep
working (Pydantic ignores unknown fields) but won't see the new column. If you
remove a column the bench depends on, you must either:
1. Bump the snapshot and rebuild against the new commit, or
2. Keep the column in production until the bench is retired

## How the bench reads the snapshot

`src/bench/snapshot.py` is the single source of truth for snapshot paths.
All bench code imports from there:

```python
from bench.snapshot import (
    SNAPSHOT_ID,
    CASE_KB_COMMIT,
    SNAPSHOT_LANCEDB_PATH,
    SNAPSHOT_CHUNKS_PARQUET,
    SNAPSHOT_SEARCH_DUCKDB,
    SNAPSHOT_FACTS_JSONL,
    assert_snapshot_present,
)
```

Updated to point at the snapshot:
- `src/bench/io_lance.py` — `PROD_LANCEDB_PATH = SNAPSHOT_LANCEDB_PATH`
- `runners/eval_quality.py` — `PROD_DUCKDB = SNAPSHOT_SEARCH_DUCKDB`
- `runners/build_adversarial.py` — `FACTS_PATH = SNAPSHOT_FACTS_JSONL`

Runners may call `assert_snapshot_present()` at startup to fail fast if any
expected file is missing.

## Re-snapshotting — when and how

You need a new snapshot when:
- Production `core/schemas.py:ChunkRecord` changes in a way the bench needs to see
- The corpus grows significantly (>20% new chunks) and you want to re-tune stratification
- A new embedding provider is added that you want to baseline
- A new source type is added (would change the source-type distribution in eval)

To create a new snapshot:

```bash
# 1. Pick the new anchor commit (must be HEAD of main)
NEW_ANCHOR=$(git -C /mnt/i/dev/Legal/case_kb rev-parse --short HEAD)
NEW_DATE=$(date +%Y-%m-%d)
NEW_ID="${NEW_DATE}_${NEW_ANCHOR}"

# 2. Copy production data into new snapshot dirs
SNAP_DATA="/mnt/i/dev/Legal/data/evidence_lake/snapshots/bench_${NEW_ID}"
SNAP_LANCE="/home/tommy/evidence_lake_indexes/bench_snapshots/${NEW_ID}"
mkdir -p "$SNAP_DATA" "$SNAP_LANCE"
cp -r /home/tommy/evidence_lake_indexes/lancedb_gemini "$SNAP_LANCE/lancedb_gemini"
cp /mnt/i/dev/Legal/data/evidence_lake/derived/parquet/chunks.parquet "$SNAP_DATA/"
cp /mnt/i/dev/Legal/data/evidence_lake/derived/indexes/search.duckdb "$SNAP_DATA/"
cp /mnt/i/dev/Legal/data/evidence_lake/derived/agent_verified_facts.jsonl "$SNAP_DATA/"
ln -s "$SNAP_LANCE/lancedb_gemini" "$SNAP_DATA/lancedb_gemini"

# 3. Update src/bench/snapshot.py with the new SNAPSHOT_ID and CASE_KB_COMMIT
# 4. Copy core/schemas.py:ChunkRecord into src/bench/schemas.py if it changed
# 5. Update this SNAPSHOT.md with new sizes, counts, and SHA-256 hashes
# 6. Commit
```

## Do NOT delete old snapshots

Old snapshots are valuable for re-running historical bench analyses. They live
outside the repo (in the data knowledge base) and don't bloat git. Disk cost is
~5.6 GB per snapshot; cheap relative to insight.
