"""Build the stratified 50K corpus sample from production LanceDB.

Outputs: bench_embeddings/data/chunk_samples/stratified_50k.parquet
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.io_lance import read_prod_chunks
from bench.sampling import stratified_sample, SAMPLE_QUOTAS

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_samples" / "stratified_50k.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    log.info("Reading production LanceDB chunks...")
    chunks = read_prod_chunks()
    log.info(f"  loaded {len(chunks):,} chunks")
    log.info(f"  source_type counts:\n{chunks['source_type'].value_counts().to_string()}")

    log.info("Stratifying sample...")
    sample = stratified_sample(chunks, seed=args.seed)
    log.info(f"  sampled {len(sample):,} chunks")
    log.info(f"  per-source: {sample['source_type'].value_counts().to_dict()}")
    log.info(f"  per-length: {sample['length_bucket'].value_counts().to_dict()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out, index=False)
    log.info(f"Wrote {args.out} ({args.out.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
