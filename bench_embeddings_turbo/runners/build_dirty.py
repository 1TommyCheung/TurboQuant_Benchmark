"""Build the dirty-corpus stratum: 500 chunks from OCR'd court PDFs + Vision-captioned photos.

The benchmark scores these separately (not folded into the main quality score).

Output: data/eval_queries/dirty_corpus.jsonl
Each record points to a chunk plus a synthetic query, with a `dirty_reason` tag.
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.perturbations import inject_typo

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_samples" / "stratified_50k.parquet"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_queries" / "dirty_corpus.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _has_ocr_artifacts(text: str) -> bool:
    """Heuristic: mid-word breaks, weird unicode, repeated whitespace."""
    if "  " in text or " \n" in text:
        return True
    if any(c in text for c in ("¶", "ﬁ", "ﬂ", "­")):
        return True
    # Mid-word hyphen breaks
    if "-\n" in text or "- " in text:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sample = pd.read_parquet(SAMPLE_PATH)
    rng = random.Random(args.seed)

    # Bucket A: OCR-noisy court_doc or solicitor_letter
    ocr_pool = sample[
        sample["source_type"].isin(["court_doc", "solicitor_letter"])
        & sample["chunk_text"].apply(_has_ocr_artifacts)
    ]
    # Bucket B: Vision-captioned photo or video
    vision_pool = sample[sample["source_type"].isin(["photo", "video"])]

    n_ocr = min(args.n // 2, len(ocr_pool))
    n_vision = min(args.n - n_ocr, len(vision_pool))

    ocr_sel = ocr_pool.sample(n=n_ocr, random_state=args.seed)
    vision_sel = vision_pool.sample(n=n_vision, random_state=args.seed)

    out: list[dict] = []
    for _, row in ocr_sel.iterrows():
        # Synthetic query: first 8 words of chunk + 1-char typo
        words = row["chunk_text"].split()[:8]
        q = inject_typo(" ".join(words), seed=rng.randint(0, 1_000_000))
        out.append({
            "id": f"DIRTY-OCR-{row['chunk_id']}",
            "query": q,
            "source_chunk_id": row["chunk_id"],
            "source_type": row["source_type"],
            "dirty_reason": "ocr_artifacts",
        })

    for _, row in vision_sel.iterrows():
        words = row["chunk_text"].split()[:8]
        q = " ".join(words)
        out.append({
            "id": f"DIRTY-VISION-{row['chunk_id']}",
            "query": q,
            "source_chunk_id": row["chunk_id"],
            "source_type": row["source_type"],
            "dirty_reason": "vision_caption",
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    log.info(f"Wrote {len(out)} dirty queries ({n_ocr} OCR + {n_vision} vision)")


if __name__ == "__main__":
    main()
