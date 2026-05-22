"""Tests for source_weights.py — weighted cited-evidence overlap per spec §5.3."""
from __future__ import annotations


def test_weights_match_spec():
    from bench.source_weights import SOURCE_TYPE_WEIGHTS
    assert SOURCE_TYPE_WEIGHTS["solicitor_letter"] == 2.0
    assert SOURCE_TYPE_WEIGHTS["court_doc"] == 2.0
    assert SOURCE_TYPE_WEIGHTS["email"] == 1.5
    assert SOURCE_TYPE_WEIGHTS["document_exchange"] == 1.5
    assert SOURCE_TYPE_WEIGHTS["financial"] == 1.5
    assert SOURCE_TYPE_WEIGHTS["whatsapp"] == 1.0
    assert SOURCE_TYPE_WEIGHTS["photo"] == 1.0


def test_weight_for_known_source():
    from bench.source_weights import weight_for
    assert weight_for("solicitor_letter") == 2.0


def test_weight_for_unknown_source_defaults_to_one():
    from bench.source_weights import weight_for
    assert weight_for("video") == 1.0
    assert weight_for("screenshot") == 1.0


def test_weighted_overlap_perfect():
    from bench.source_weights import weighted_cited_overlap
    cited_with_type = [("cid-1", "solicitor_letter"), ("cid-2", "email")]
    returned = {"cid-1", "cid-2"}
    assert weighted_cited_overlap(cited_with_type, returned) == 1.0


def test_weighted_overlap_missed_high_weight():
    from bench.source_weights import weighted_cited_overlap
    cited_a = [("cid-1", "solicitor_letter"), ("cid-2", "whatsapp")]
    surfaced_a = {"cid-1"}  # surfaced solicitor_letter, missed whatsapp
    surfaced_b = {"cid-2"}  # surfaced whatsapp, missed solicitor_letter
    assert weighted_cited_overlap(cited_a, surfaced_a) > weighted_cited_overlap(cited_a, surfaced_b)


def test_weighted_overlap_empty_cited_returns_one():
    from bench.source_weights import weighted_cited_overlap
    assert weighted_cited_overlap([], {"anything"}) == 1.0
