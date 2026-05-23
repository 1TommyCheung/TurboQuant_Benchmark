"""Read-only chunk snapshot access, write paths for the local bench.

The snapshot (defined in `bench.snapshot`) freezes production data at a
specific case_kb commit so future pipeline iterations don't invalidate
historical bench runs. See SNAPSHOT.md.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .snapshot import SNAPSHOT_CHUNKS_PARQUET, SNAPSHOT_LANCEDB_PATH

# Alias preserved for back-compat with existing runners.
PROD_LANCEDB_PATH = SNAPSHOT_LANCEDB_PATH
BENCH_LANCEDB_ROOT = Path(__file__).resolve().parents[1] / "indexes"

# Columns to project. The `vector` column is intentionally omitted to
# keep DataFrame memory bounded; load it separately if needed.
_PROD_CHUNK_COLS = [
    "chunk_id", "evidence_id", "source_type", "chunk_text",
    "context_header", "token_count", "party_from", "date_sgt",
    "legal_issues", "is_privileged", "is_wp", "in_scope",
]


def read_prod_chunks() -> pd.DataFrame:
    """Read frozen production chunks from parquet as a DataFrame.

    The original benchmark read this metadata from a WSL-only LanceDB snapshot.
    This Windows-native TurboQuant copy uses the read-only parquet snapshot
    instead so all generated LanceDB tables can stay under this experiment dir.
    """
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(SNAPSHOT_CHUNKS_PARQUET).names)
    cols = [c for c in _PROD_CHUNK_COLS if c in schema_names]
    df = pd.read_parquet(SNAPSHOT_CHUNKS_PARQUET, columns=cols)
    for col in _PROD_CHUNK_COLS:
        if col not in df.columns:
            df[col] = None
    return df[_PROD_CHUNK_COLS]


def bench_lancedb_path(model_id: str) -> Path:
    """Return path for a model's bench LanceDB table."""
    BENCH_LANCEDB_ROOT.mkdir(parents=True, exist_ok=True)
    return BENCH_LANCEDB_ROOT / f"{model_id}.lance"
