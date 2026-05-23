"""Retrieval metrics: recall@k, MRR, NDCG, paired bootstrap CIs."""
from __future__ import annotations
import math
from typing import Iterable
import numpy as np


def recall_at_k(ranked: list[str], positives: set[str] | Iterable[str], k: int) -> float:
    """Fraction of positives found in top-k ranked list."""
    positives = set(positives)
    if not positives:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for d in topk if d in positives)
    return hits / len(positives)


def mrr_at_k(ranked: list[str], positives: set[str] | Iterable[str], k: int) -> float:
    """Reciprocal rank of the first relevant hit in top-k, 0 if none."""
    positives = set(positives)
    for i, d in enumerate(ranked[:k], start=1):
        if d in positives:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    """Normalised DCG with binary log2(rank+1) discount.

    grades: dict mapping doc_id → graded relevance score (0, 1, 2, 3).
    """
    def dcg(items: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(items))

    ranked_grades = [grades.get(d, 0) for d in ranked[:k]]
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = dcg(ideal_grades)
    if idcg == 0:
        return 0.0
    return dcg(ranked_grades) / idcg


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of values."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi
