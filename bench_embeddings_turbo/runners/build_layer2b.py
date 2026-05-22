"""Generate Layer 2b: LLM-synthetic queries with self-leakage filter.

For each sampled chunk:
1. Prompt Gemini 2.5 flash-lite (70%) or Claude Sonnet (30%) for a natural query.
2. Embed query + chunk with a held-out model (bge-small) for cosine check.
3. Drop if 4-gram overlap > 0.6 OR cosine > 0.92.

Target output: ~1000 queries (after filtering).
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import litellm
from litellm import completion
from sentence_transformers import SentenceTransformer

# Gemini and Anthropic don't accept `seed`; drop unsupported params silently.
litellm.drop_params = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.leakage import is_leaky

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_samples" / "stratified_50k.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are simulating a Singapore Family Court litigant searching through their personal evidence archive. Given the following piece of evidence, write ONE natural-sounding search query (8-20 words) that the litigant might type to find this exact piece of evidence later. The query should:

- Be casual and natural (NOT a paraphrase of the evidence)
- Mention parties or dates if relevant
- Sometimes include typos, abbreviations, or Singlish particles
- NOT directly quote the evidence

Evidence (source_type={source_type}, date={date}, party_from={party}):
\"\"\"
{chunk_text}
\"\"\"

Return ONLY the search query, nothing else."""


def generate_one(chunk: dict, model: str, seed: int) -> str | None:
    """Call the LLM to generate one query. Return None on failure."""
    prompt = PROMPT_TEMPLATE.format(
        source_type=chunk["source_type"],
        date=chunk.get("date_sgt", "unknown"),
        party=chunk.get("party_from", "unknown"),
        chunk_text=chunk["chunk_text"][:2000],
    )
    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            seed=seed,
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip().strip('"').strip()
    except Exception as e:
        log.warning(f"  LLM failed: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=1500, help="Chunks to sample before filtering")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gemini-model", default="gemini/gemini-2.5-flash-lite")
    ap.add_argument("--claude-model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--claude-fraction", type=float, default=0.3)
    args = ap.parse_args()

    log.info("Loading 50K corpus sample...")
    sample = pd.read_parquet(SAMPLE_PATH)
    rng = random.Random(args.seed)

    # Stratified sub-sample for Layer 2b
    chunks = sample.sample(n=min(args.n_chunks, len(sample)), random_state=args.seed)

    log.info("Loading bge-small for leakage check (held-out model)...")
    holdout = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

    out: list[dict] = []
    dropped_ngram = 0
    dropped_cosine = 0
    for i, (_, row) in enumerate(chunks.iterrows()):
        use_claude = rng.random() < args.claude_fraction
        model = args.claude_model if use_claude else args.gemini_model
        query = generate_one(row.to_dict(), model=model, seed=args.seed + i)
        if not query or len(query) < 4:
            continue

        # Held-out leakage check
        q_emb, c_emb = holdout.encode([query, row["chunk_text"][:2000]], normalize_embeddings=True)
        q_emb, c_emb = np.array(q_emb), np.array(c_emb)

        leaky = is_leaky(query, row["chunk_text"], q_emb, c_emb)
        if leaky:
            from bench.leakage import ngram_overlap, cosine
            if ngram_overlap(query, row["chunk_text"]) > 0.6:
                dropped_ngram += 1
            else:
                dropped_cosine += 1
            continue

        out.append({
            "id": f"L2b-{i:05d}",
            "query": query,
            "source_chunk_id": row["chunk_id"],
            "source_type": row["source_type"],
            "length_bucket": row["length_bucket"],
            "generator": "claude" if use_claude else "gemini",
        })
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{len(chunks)}: kept {len(out)}, dropped {dropped_ngram+dropped_cosine}")

    out_path = DATA / "layer2b_synthetic.jsonl"
    with out_path.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    log.info(f"Wrote {len(out)} queries to {out_path}")
    log.info(f"Dropped: ngram={dropped_ngram} cosine={dropped_cosine}")


if __name__ == "__main__":
    main()
