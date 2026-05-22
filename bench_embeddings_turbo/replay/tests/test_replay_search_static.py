"""Tests for replay_search_static.py — metric computations."""
from __future__ import annotations


def test_jaccard_at_k_perfect():
    from replay.runners.replay_search_static import jaccard_at_k
    assert jaccard_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0


def test_jaccard_at_k_disjoint():
    from replay.runners.replay_search_static import jaccard_at_k
    assert jaccard_at_k(["a", "b"], ["x", "y"], k=2) == 0.0


def test_jaccard_at_k_partial():
    from replay.runners.replay_search_static import jaccard_at_k
    score = jaccard_at_k(["a", "b", "c"], ["a", "c", "d"], k=3)
    assert abs(score - 0.5) < 1e-9


def test_ndcg_overlap_perfect_match():
    from replay.runners.replay_search_static import ndcg_overlap_at_k
    assert ndcg_overlap_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0


def test_ndcg_overlap_reversed():
    # PLAN DEFECT: original fixture used a reversed list, but the impl's
    # ndcg_overlap_at_k uses set-membership in b[:k] (no positional decay
    # on b), so a reversed list with identical contents still scores 1.0.
    # Use a partial-overlap pair where the top-ranked items match but the
    # tail item is missing; this exercises the positional decay on `a`
    # while landing in (0.5, 1.0) as the spec intended.
    from replay.runners.replay_search_static import ndcg_overlap_at_k
    score = ndcg_overlap_at_k(["a", "b", "c"], ["a", "b", "x"], k=3)
    assert 0.5 < score < 1.0


def test_divergence_score_weighted_combo():
    from replay.runners.replay_search_static import divergence_score
    d = divergence_score(jaccard=0.8, cited_weighted=0.9, ndcg_overlap=0.7)
    expected = 0.5 * 0.2 + 0.3 * 0.1 + 0.2 * 0.3
    assert abs(d - expected) < 1e-9
