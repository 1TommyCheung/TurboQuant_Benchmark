"""Shared pytest fixtures for Phase A tests."""
from __future__ import annotations
from pathlib import Path
import pytest


@pytest.fixture
def replay_root() -> Path:
    """Absolute path to bench_embeddings/replay/."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def case_kb_root(replay_root: Path) -> Path:
    return replay_root.parent.parent


@pytest.fixture
def sample_session_html(case_kb_root: Path) -> Path:
    """Path to the pi-session HTML used as input."""
    return case_kb_root / "agent" / "pi-session-2026-04-01T05-26-01-675Z_ccf94dfb-f6d2-48f3-b40f-ef534a97268a.html"
