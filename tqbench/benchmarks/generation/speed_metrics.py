"""Aggregation functions for speed benchmark results."""
from __future__ import annotations
import numpy as np

from tqbench.benchmarks.generation.clients import StreamResult


def aggregate_stream_results(
    results: list[StreamResult],
    wall_time_s: float,
) -> dict:
    ttfts = np.array([r.ttft_s for r in results])
    all_itl = np.concatenate([np.array(r.itl_ms) for r in results if r.itl_ms])
    latencies = np.array([r.total_time_s for r in results])
    total_tokens = sum(r.completion_tokens for r in results)

    return {
        "n_requests": len(results),
        "total_output_tokens": int(total_tokens),
        "wall_time_s": wall_time_s,
        "throughput_tok_s": total_tokens / wall_time_s if wall_time_s > 0 else 0,
        "ttft_median_s": float(np.median(ttfts)),
        "ttft_p95_s": float(np.percentile(ttfts, 95)),
        "ttft_p99_s": float(np.percentile(ttfts, 99)),
        "itl_median_ms": float(np.median(all_itl)) if len(all_itl) > 0 else 0,
        "itl_p95_ms": float(np.percentile(all_itl, 95)) if len(all_itl) > 0 else 0,
        "latency_median_s": float(np.median(latencies)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
    }


def aggregate_ttft_results(ttfts: list[float]) -> dict:
    arr = np.array(ttfts)
    return {
        "n": len(arr),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "mean_s": float(np.mean(arr)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
    }
