"""Tests for retrieval metrics + bootstrap CIs."""
from __future__ import annotations
import numpy as np
from bench.metrics import recall_at_k, mrr_at_k, ndcg_at_k, bootstrap_ci


def test_recall_at_k_all_in_topk():
    ranked = ["a", "b", "c", "d", "e"]
    positives = {"a", "c"}
    assert recall_at_k(ranked, positives, k=5) == 1.0


def test_recall_at_k_partial():
    ranked = ["x", "a", "y", "z"]
    positives = {"a", "b"}
    assert recall_at_k(ranked, positives, k=4) == 0.5


def test_recall_at_k_none():
    ranked = ["x", "y", "z"]
    positives = {"a", "b"}
    assert recall_at_k(ranked, positives, k=3) == 0.0


def test_mrr_at_k_first_hit():
    ranked = ["x", "a", "b"]
    positives = {"a"}
    assert mrr_at_k(ranked, positives, k=3) == 0.5  # rank 2


def test_mrr_at_k_no_hit_returns_zero():
    ranked = ["x", "y", "z"]
    positives = {"a"}
    assert mrr_at_k(ranked, positives, k=3) == 0.0


def test_ndcg_at_k_perfect_ranking():
    ranked = ["a", "b", "c"]
    grades = {"a": 3, "b": 2, "c": 1}
    score = ndcg_at_k(ranked, grades, k=3)
    assert score == 1.0


def test_ndcg_at_k_reverse_ranking():
    ranked = ["c", "b", "a"]
    grades = {"a": 3, "b": 2, "c": 1}
    score = ndcg_at_k(ranked, grades, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_all_zero_grades():
    ranked = ["a", "b"]
    grades = {"a": 0, "b": 0}
    score = ndcg_at_k(ranked, grades, k=2)
    assert score == 0.0


def test_bootstrap_ci_returns_interval():
    rng = np.random.default_rng(42)
    scores = rng.normal(0.5, 0.1, size=200)
    lo, hi = bootstrap_ci(scores, n_resamples=500, seed=42)
    assert lo < hi
    assert 0.3 < lo < 0.6
    assert 0.4 < hi < 0.7


def test_bootstrap_ci_paired_difference():
    rng = np.random.default_rng(42)
    a = rng.normal(0.7, 0.1, size=200)
    b = rng.normal(0.5, 0.1, size=200)
    delta = a - b
    lo, hi = bootstrap_ci(delta, n_resamples=500, seed=42)
    # 95% CI for delta should not contain 0
    assert lo > 0
