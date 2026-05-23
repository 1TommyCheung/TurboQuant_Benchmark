from __future__ import annotations
from tqbench.benchmarks import discover_benchmarks


def test_discover_finds_embeddings():
    benchmarks = discover_benchmarks()
    assert "embeddings" in benchmarks
    assert benchmarks["embeddings"]["description"]
    assert benchmarks["embeddings"]["entry"]
