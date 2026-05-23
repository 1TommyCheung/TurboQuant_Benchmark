from __future__ import annotations
import pytest
import numpy as np
from tqbench.benchmarks.generation.speed_metrics import (
    aggregate_stream_results, aggregate_ttft_results,
)
from tqbench.benchmarks.generation.clients import StreamResult


def _stream(ttft: float, itl: list[float], comp_tokens: int = 10) -> StreamResult:
    return StreamResult(
        text="x" * comp_tokens,
        prompt_tokens=50,
        completion_tokens=comp_tokens,
        ttft_s=ttft,
        itl_ms=itl,
        total_time_s=ttft + sum(itl) / 1000,
    )


def test_aggregate_stream_results():
    results = [
        _stream(0.1, [20.0, 25.0, 30.0]),
        _stream(0.2, [22.0, 28.0, 35.0]),
        _stream(0.15, [21.0, 26.0, 32.0]),
    ]
    agg = aggregate_stream_results(results, wall_time_s=1.0)
    assert "ttft_median_s" in agg
    assert "ttft_p95_s" in agg
    assert "ttft_p99_s" in agg
    assert "itl_median_ms" in agg
    assert "itl_p95_ms" in agg
    assert "throughput_tok_s" in agg
    assert "latency_median_s" in agg
    assert "latency_p95_s" in agg
    assert agg["n_requests"] == 3
    assert agg["throughput_tok_s"] > 0
    assert agg["ttft_median_s"] == pytest.approx(0.15, abs=0.01)


def test_aggregate_ttft_results():
    ttfts = [0.5, 0.6, 0.55, 0.52, 0.58, 0.51, 0.53, 0.57, 0.54, 0.56]
    agg = aggregate_ttft_results(ttfts)
    assert "median_s" in agg
    assert "p95_s" in agg
    assert agg["n"] == 10
    assert 0.5 < agg["median_s"] < 0.6
