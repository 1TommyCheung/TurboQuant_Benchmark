"""Tests for synthetic-query self-leakage filter."""
from __future__ import annotations
import numpy as np
from bench.leakage import ngram_overlap, is_leaky, MAX_NGRAM_OVERLAP


def test_ngram_overlap_perfect():
    s = "the quick brown fox jumps over the lazy dog"
    assert ngram_overlap(s, s, n=4) == 1.0


def test_ngram_overlap_disjoint():
    a = "the quick brown fox"
    b = "completely different sentence here"
    assert ngram_overlap(a, b, n=4) == 0.0


def test_ngram_overlap_partial():
    a = "the quick brown fox jumps over the lazy dog"
    b = "the quick brown fox runs under the lazy cat"
    o = ngram_overlap(a, b, n=4)
    assert 0.0 < o < 1.0


def test_is_leaky_high_ngram_overlap():
    query = "tommy sold his hdb flat in 2013 for proceeds of sale used towards"
    chunk = "tommy sold his hdb flat in 2013 for proceeds of sale used towards the new property purchase"
    # Force low cosine to isolate n-gram path
    cos = np.array([0.0])
    assert is_leaky(query, chunk, query_embed=cos, chunk_embed=cos)


def test_is_leaky_high_cosine():
    query = "did the kids go to school yesterday"
    chunk = "school attendance was normal on the prior day"
    # Force low n-gram, but high cosine
    qe = np.array([1.0, 0.0])
    ce = np.array([0.95, 0.1])
    qe = qe / np.linalg.norm(qe)
    ce = ce / np.linalg.norm(ce)
    assert is_leaky(query, chunk, query_embed=qe, chunk_embed=ce)


def test_is_leaky_safe_query():
    query = "find emails about valley point in december"
    chunk = "the meeting was held at a different location entirely"
    qe = np.array([1.0, 0.0])
    ce = np.array([0.0, 1.0])
    assert not is_leaky(query, chunk, query_embed=qe, chunk_embed=ce)


def test_max_ngram_overlap_constant():
    assert MAX_NGRAM_OVERLAP == 0.6
