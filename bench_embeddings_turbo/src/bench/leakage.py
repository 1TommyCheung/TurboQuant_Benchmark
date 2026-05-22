"""Self-leakage filter for synthetic queries (Layer 2b).

A query is "leaky" if it is too similar to its source chunk —
either lexically (high n-gram overlap) or semantically (high cosine
under a held-out embedding model). Leaky queries trivialise retrieval
because the model just has to find a paraphrase of itself.
"""
from __future__ import annotations
import re
import numpy as np

MAX_NGRAM_OVERLAP: float = 0.6   # spec §6: drop if 4-gram overlap > 0.6
MAX_COSINE: float = 0.92         # spec §6: drop if held-out cosine > 0.92


def _tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(a: str, b: str, n: int = 4) -> float:
    """Jaccard overlap of n-grams between two strings."""
    A = _ngrams(_tokenize(a), n)
    B = _ngrams(_tokenize(b), n)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two unit-or-not vectors."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def is_leaky(
    query: str,
    chunk_text: str,
    query_embed: np.ndarray,
    chunk_embed: np.ndarray,
    max_ngram_overlap: float = MAX_NGRAM_OVERLAP,
    max_cosine: float = MAX_COSINE,
) -> bool:
    """Return True if query is too similar to source chunk (leaky)."""
    if ngram_overlap(query, chunk_text) > max_ngram_overlap:
        return True
    if cosine(query_embed, chunk_embed) > max_cosine:
        return True
    return False
