"""Tests for stratified sampler."""
from __future__ import annotations
import pandas as pd
import pytest
from bench.sampling import stratified_sample, length_bucket, SAMPLE_QUOTAS


def _fake_chunks(n_per_type: dict[str, int]) -> pd.DataFrame:
    rows = []
    for source_type, n in n_per_type.items():
        for i in range(n):
            rows.append({
                "chunk_id": f"{source_type}-{i}",
                "source_type": source_type,
                "chunk_text": "word " * (50 if i % 4 == 0 else 600 if i % 4 == 1 else 3000 if i % 4 == 2 else 9000),
                "token_count": 50 if i % 4 == 0 else 600 if i % 4 == 1 else 3000 if i % 4 == 2 else 9000,
            })
    return pd.DataFrame(rows)


def test_quotas_sum_to_50000():
    assert sum(SAMPLE_QUOTAS.values()) == 50_000


def test_length_bucket_boundaries():
    assert length_bucket(100) == "short"
    assert length_bucket(511) == "short"
    assert length_bucket(512) == "medium"
    assert length_bucket(2047) == "medium"
    assert length_bucket(2048) == "long"
    assert length_bucket(8191) == "long"
    assert length_bucket(8192) == "very_long"
    assert length_bucket(50000) == "very_long"


def test_stratified_sample_respects_quotas():
    df = _fake_chunks({"email": 30_000, "whatsapp": 30_000, "court_doc": 10_000,
                       "solicitor_letter": 3_000, "photo": 3_000, "video": 1_000,
                       "document_exchange": 2_000, "financial": 500})
    sample = stratified_sample(df, seed=42)
    counts = sample["source_type"].value_counts().to_dict()
    assert counts.get("email") == 18_500
    assert counts.get("whatsapp") == 18_000
    assert counts.get("court_doc") == 7_500
    assert counts.get("solicitor_letter") == 2_400
    assert counts.get("photo", 0) + counts.get("video", 0) == 2_000
    assert counts.get("document_exchange") == 1_300
    assert counts.get("financial") == 300
    assert len(sample) == 50_000


def test_stratified_sample_deterministic():
    df = _fake_chunks({"email": 25_000, "whatsapp": 25_000, "court_doc": 8_000,
                       "solicitor_letter": 5_000, "photo": 3_000, "video": 1_000,
                       "document_exchange": 1_500, "financial": 1_500})
    s1 = stratified_sample(df, seed=42)
    s2 = stratified_sample(df, seed=42)
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))


def test_stratified_sample_raises_on_insufficient_data():
    df = _fake_chunks({"email": 100, "whatsapp": 100, "court_doc": 100,
                       "solicitor_letter": 100, "photo": 100, "video": 100,
                       "document_exchange": 100, "financial": 100})
    with pytest.raises(ValueError, match="insufficient"):
        stratified_sample(df, seed=42)


def test_length_bucket_column_added():
    df = _fake_chunks({"email": 20_000, "whatsapp": 20_000, "court_doc": 8_000,
                       "solicitor_letter": 5_000, "photo": 2_000, "video": 1_000,
                       "document_exchange": 1_500, "financial": 1_500})
    sample = stratified_sample(df, seed=42)
    assert "length_bucket" in sample.columns
    assert set(sample["length_bucket"].unique()) <= {"short", "medium", "long", "very_long"}
