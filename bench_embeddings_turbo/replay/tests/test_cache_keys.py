"""Tests for update_cache_keys.py — SHA-keyed cache invalidation tracker."""
from __future__ import annotations
import json


def test_sha_of_file_content(tmp_path):
    from replay.runners.update_cache_keys import _sha_of_file
    p = tmp_path / "x.txt"
    p.write_text("hello")
    assert len(_sha_of_file(p)) == 64
    q = tmp_path / "y.txt"
    q.write_text("hello")
    assert _sha_of_file(p) == _sha_of_file(q)


def test_sha_of_missing_file_returns_none(tmp_path):
    from replay.runners.update_cache_keys import _sha_of_file
    assert _sha_of_file(tmp_path / "nope.txt") is None


def test_compute_cache_keys_has_all_10_keys():
    """Snapshot-aligned keys per spec §3.3 (extended)."""
    from replay.runners.update_cache_keys import compute_cache_keys
    keys = compute_cache_keys()
    expected = {
        "snapshot_id",
        "case_kb_commit",
        "snapshot_chunks_parquet_sha",
        "snapshot_search_duckdb_sha",
        "snapshot_facts_jsonl_sha",
        "snapshot_lancedb_dir_sha",
        "bench_snapshot_py_sha",
        "bench_schemas_py_sha",
        "bench_scoring_py_sha",
        "bench_source_weights_py_sha",
    }
    assert set(keys.keys()) == expected


def test_snapshot_id_is_pinned_constant():
    from replay.runners.update_cache_keys import compute_cache_keys
    keys = compute_cache_keys()
    assert keys["snapshot_id"] == "2026-05-16_1fe458f"
    assert keys["case_kb_commit"] == "1fe458fa6"


def test_dir_sha_changes_when_any_file_changes(tmp_path):
    from replay.runners.update_cache_keys import _sha_of_dir
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    sha_1 = _sha_of_dir(tmp_path)
    (tmp_path / "b.txt").write_text("WORLD")
    sha_2 = _sha_of_dir(tmp_path)
    assert sha_1 != sha_2
