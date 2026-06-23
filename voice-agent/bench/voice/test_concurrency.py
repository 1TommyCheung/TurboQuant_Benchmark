"""T05 concurrent sessions load tests for the audio service.

Pass criterion (Pattern C design): p95 TTFTAudio < 1600 ms for 5 simultaneous
T02-style sessions (~2x solo budget).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness import VoiceSession
from metrics import p95, summarize


async def _run_one(ws_url: str, wav: Path) -> float:
    async with VoiceSession(ws_url) as session:
        await session.send_wav(wav)
        await session.wait_for_event("tts_first_audio_sample", timeout=20.0)
        return session.compute_ttft_audio_ms()


@pytest.mark.asyncio
@pytest.mark.wav_fixture("medium_time.wav")
@pytest.mark.parametrize("concurrency", [1, 2, 5], ids=lambda n: f"concurrent_{n}")
async def test_concurrent_sessions(
    concurrency: int,
    ws_url: str,
    fixture_dir: Path,
    record_property,
) -> None:
    wav = fixture_dir / "medium_time.wav"
    results = await asyncio.gather(*[_run_one(ws_url, wav) for _ in range(concurrency)])
    stats = summarize(results)
    record_property("concurrency", concurrency)
    record_property("ttft_audio_ms", results)
    record_property("summary", stats)

    if concurrency >= 5:
        # T05 explicit gate
        p95_ms = p95(results)
        assert p95_ms < 1600.0, f"T05_concurrent_5: p95 TTFTAudio {p95_ms:.1f}ms >= 1600ms"
