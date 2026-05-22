"""Phase 1: embed the 50K corpus sample with one model.

For each chunk in stratified_50k.parquet:
- Run encoder.encode(chunk_text) → dim-d vector
- Write to indexes/{model_id}.lance with chunk_id, evidence_id,
  source_type, length_bucket, vector.

Run once per model_id.

Usage:
    python -m runners.embed_corpus --model qwen3-embedding-8b-int8
"""
from __future__ import annotations
import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.models import get_candidate, load_embedder
from bench.io_lance import bench_lancedb_path

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_samples" / "stratified_50k.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_id from config/models.yaml")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, help="embed only the first N sampled chunks")
    args = ap.parse_args()

    spec = get_candidate(args.model)
    out_path = bench_lancedb_path(spec.id)
    if out_path.exists() and not args.overwrite:
        log.info(f"Index exists at {out_path}, skipping (use --overwrite to force)")
        return
    if out_path.exists() and args.overwrite:
        if out_path.is_symlink():
            # Preserve symlink (target is on a different filesystem); clear its contents.
            for child in out_path.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            shutil.rmtree(out_path)

    log.info(f"Loading {len(spec.id)}-char model id; spec={spec}")
    embedder = load_embedder(spec.id)
    log.info(f"Loading corpus from {SAMPLE_PATH}")
    sample = pd.read_parquet(SAMPLE_PATH)
    if args.limit:
        sample = sample.head(args.limit)
    log.info(f"  {len(sample):,} chunks to embed")

    schema = pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("evidence_id", pa.string()),
        pa.field("source_type", pa.string()),
        pa.field("length_bucket", pa.string()),
        pa.field("token_count", pa.int64()),
        pa.field("date_sgt", pa.string()),
        pa.field("party_from", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), spec.dim)),
    ])

    db = lancedb.connect(str(out_path.parent))
    tbl = db.create_table(spec.id, schema=schema, exist_ok=True)

    start = time.time()
    total = len(sample)
    for i in range(0, total, args.batch_size * 10):
        batch = sample.iloc[i:i + args.batch_size * 10]
        texts = batch["chunk_text"].tolist()
        vecs = embedder.encode(texts, batch_size=args.batch_size)
        if np.isnan(vecs).any():
            raise RuntimeError(f"NaN vectors in batch starting at {i}")
        rows = []
        for j, (_, row) in enumerate(batch.iterrows()):
            rows.append({
                "chunk_id": row["chunk_id"],
                "evidence_id": row["evidence_id"],
                "source_type": row["source_type"],
                "length_bucket": row["length_bucket"],
                "token_count": int(row["token_count"]),
                "date_sgt": str(row.get("date_sgt", "")),
                "party_from": str(row.get("party_from", "")),
                "vector": vecs[j].astype("float32").tolist(),
            })
        tbl.add(rows)
        elapsed = time.time() - start
        rate = (i + len(batch)) / elapsed
        eta = (total - i - len(batch)) / rate if rate > 0 else 0
        log.info(f"  {i + len(batch):,}/{total:,}  rate={rate:.1f} chunks/s  eta={eta/60:.1f} min")

    log.info(f"Done. Wrote {tbl.count_rows()} rows to {out_path}")


if __name__ == "__main__":
    main()
