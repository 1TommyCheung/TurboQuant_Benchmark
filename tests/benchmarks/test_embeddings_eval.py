from __future__ import annotations
import numpy as np
from tqbench.benchmarks.embeddings.eval.leakage import ngram_overlap, cosine, is_leaky
from tqbench.benchmarks.embeddings.eval.perturbations import inject_typo, perturb_all
from tqbench.benchmarks.embeddings.eval.stack import rrf_fuse
from tqbench.benchmarks.embeddings.eval.source_weights import weight_for, weighted_cited_overlap


def test_ngram_overlap_identical():
    s = "the quick brown fox jumps over the lazy dog"
    assert ngram_overlap(s, s) == 1.0


def test_ngram_overlap_disjoint():
    assert ngram_overlap("alpha beta gamma delta", "one two three four") == 0.0


def test_cosine_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine(v, v) == 1.0


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine(a, b) == 0.0


def test_is_leaky_high_overlap():
    text = "the quick brown fox jumps over the lazy dog near the river"
    v = np.array([1.0, 0.0])
    assert is_leaky(text, text, v, v)


def test_inject_typo_changes_text():
    q = "what emails did tommy send in february"
    result = inject_typo(q, seed=42)
    assert result != q


def test_perturb_all_produces_variants():
    q = "What did Lee & Lee say about the custody hearing in January 2026?"
    variants = perturb_all(q, seed=42)
    assert len(variants) >= 3
    assert all(v != q for v in variants)


def test_rrf_fuse_basic():
    list1 = ["a", "b", "c"]
    list2 = ["b", "c", "d"]
    fused = rrf_fuse([list1, list2], k=3)
    assert "b" in fused
    assert len(fused) == 3


def test_weight_for_court_doc():
    assert weight_for("court_doc") == 2.0
    assert weight_for("whatsapp") == 1.0
    assert weight_for("unknown_type") == 1.0


def test_weighted_cited_overlap_all_found():
    cited = [("c1", "court_doc"), ("c2", "email")]
    returned = {"c1", "c2", "c3"}
    assert weighted_cited_overlap(cited, returned) == 1.0


def test_weighted_cited_overlap_none_found():
    cited = [("c1", "court_doc")]
    returned = {"c99"}
    assert weighted_cited_overlap(cited, returned) == 0.0
