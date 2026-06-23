"""Shared pytest fixtures for the voice benchmark."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from harness import VoiceSession


FIXTURE_DIR = Path(__file__).parent / "audio" / "fixtures"
DEFAULT_WS_URL = os.environ.get("VOICE_WS_URL", "ws://127.0.0.1:8090/ws")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "wav_fixture(name): mark test as requiring a WAV fixture under audio/fixtures/",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests whose fixture WAV is missing (real recordings live outside git)."""
    for item in items:
        for marker in item.iter_markers(name="wav_fixture"):
            name = marker.args[0]
            path = FIXTURE_DIR / name
            if not path.exists() or path.stat().st_size == 0:
                item.add_marker(
                    pytest.mark.skip(reason=f"fixture WAV not present: {path}")
                )
                break


@pytest.fixture(scope="session")
def event_loop():  # pragma: no cover - pytest-asyncio compat
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def ws_url() -> str:
    return DEFAULT_WS_URL


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest_asyncio.fixture
async def voice_session(ws_url: str) -> AsyncIterator[VoiceSession]:
    session = VoiceSession(ws_url)
    await session.connect()
    try:
        yield session
    finally:
        await session.close()


@pytest.fixture
def audio_clock():
    """Monotonic clock for tests that need their own timing reference."""
    return time.monotonic
