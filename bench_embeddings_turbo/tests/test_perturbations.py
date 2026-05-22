"""Tests for query perturbation generators."""
from __future__ import annotations
import re
from bench.perturbations import (
    inject_typo, abbreviate_parties, fuzz_dates,
    code_switch, lowercase_proper_nouns, perturb_all,
)


def test_inject_typo_makes_single_char_change():
    q = "find email from alessandra"
    out = inject_typo(q, seed=1)
    assert out != q
    # one char different, or one inserted/deleted
    assert abs(len(out) - len(q)) <= 1


def test_inject_typo_deterministic():
    q = "find email from alessandra"
    assert inject_typo(q, seed=1) == inject_typo(q, seed=1)


def test_abbreviate_parties_lee_and_lee():
    q = "did lee & lee send a letter in february 2026"
    out = abbreviate_parties(q)
    assert "LL" in out or "ll" in out.lower()
    assert "lee & lee" not in out.lower()


def test_abbreviate_parties_gjc():
    q = "email from gloria james-civetta about property"
    out = abbreviate_parties(q)
    assert "gjc" in out.lower()


def test_fuzz_dates_replaces_month_year():
    q = "letter sent on 5 February 2026"
    out = fuzz_dates(q, seed=1)
    # Original specific date should be gone, replaced with fuzzy form
    assert "5 february 2026" not in out.lower()
    # Some fuzzy temporal hint should remain
    assert any(tok in out.lower() for tok in ["feb", "early 2026", "around", "2026"])


def test_code_switch_adds_singlish_particle():
    q = "did we talk about the children"
    out = code_switch(q, seed=1)
    assert any(p in out.lower() for p in ["lah", "liao", "got", "hor", "leh"])


def test_lowercase_proper_nouns():
    q = "Email from Lee & Lee about Tristan at Valley Point"
    out = lowercase_proper_nouns(q)
    assert "Lee" not in out
    assert "Tristan" not in out
    assert "Valley" not in out


def test_perturb_all_returns_multiple_variants():
    q = "did lee & lee send a letter on 5 february 2026 about tristan"
    variants = perturb_all(q, seed=42)
    assert len(variants) >= 5
    # All variants should be different from the original
    assert all(v != q for v in variants)
    # Each should be a non-empty string
    assert all(isinstance(v, str) and len(v) > 0 for v in variants)


def test_perturb_all_deterministic():
    q = "find letter from gjc"
    a = perturb_all(q, seed=42)
    b = perturb_all(q, seed=42)
    assert a == b
