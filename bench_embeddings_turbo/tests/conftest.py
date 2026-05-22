"""Shared pytest fixtures for bench_embeddings tests."""
from __future__ import annotations
from pathlib import Path
import pytest


@pytest.fixture
def bench_root() -> Path:
    """Absolute path to the bench_embeddings/ directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures_dir(bench_root: Path) -> Path:
    return bench_root / "tests" / "fixtures"
