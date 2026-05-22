"""Smoke tests for end-to-end retrieval stack."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from bench.stack import vector_only_retrieve, hybrid_retrieve


class _FakeIndex:
    """Stand-in LanceDB table with .search(query_vec).limit(k).to_pandas()."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def search(self, vec, vector_column_name="vector"):
        self._vec = np.asarray(vec)
        return self

    def limit(self, k):
        self._k = k
        return self

    def to_pandas(self):
        # Return df sorted by dot-product descending
        scores = np.array([np.dot(self._vec, v) for v in self.df["vector"]])
        order = np.argsort(scores)[::-1][:self._k]
        out = self.df.iloc[order].copy()
        out["_distance"] = -scores[order]
        return out


def _fake_table() -> _FakeIndex:
    return _FakeIndex(pd.DataFrame({
        "chunk_id": [f"c{i}" for i in range(5)],
        "evidence_id": [f"e{i}" for i in range(5)],
        "vector": [np.array([1.0, 0.0]), np.array([0.9, 0.1]),
                   np.array([0.0, 1.0]), np.array([-1.0, 0.0]),
                   np.array([0.5, 0.5])],
        "chunk_text": ["t" for _ in range(5)],
    }))


def test_vector_only_retrieve_returns_topk_chunk_ids():
    tbl = _fake_table()
    qvec = np.array([1.0, 0.0])
    out = vector_only_retrieve(tbl, qvec, k=3)
    assert out[0] == "c0"
    assert len(out) == 3
