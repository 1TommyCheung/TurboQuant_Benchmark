"""T01-T03 end-to-end latency tests (EOU to first audio out).

Pass criteria taken from the Pattern C design doc:
    T01_short_greeting    -> ttft_audio_ms < 600
    T02_medium_factual    -> ttft_audio_ms < 800
    T03_long_explanation  -> ttft_audio_ms < 1200
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import VoiceSession


LATENCY_CASES = [
    pytest.param(
        "T01_short_greeting", "short_hi.wav", 600.0, "hi",
        marks=pytest.mark.wav_fixture("short_hi.wav"),
        id="T01_short_greeting",
    ),
    pytest.param(
        "T02_medium_factual", "medium_time.wav", 800.0, "what time is it",
        marks=pytest.mark.wav_fixture("medium_time.wav"),
        id="T02_medium_factual",
    ),
    pytest.param(
        "T03_long_explanation", "long_explain.wav", 1200.0,
        "explain how neural networks work",
        marks=pytest.mark.wav_fixture("long_explain.wav"),
        id="T03_long_explanation",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,wav_name,max_ttft_ms,reference", LATENCY_CASES)
async def test_latency_case(
    case_id: str,
    wav_name: str,
    max_ttft_ms: float,
    reference: str,
    ws_url: str,
    fixture_dir: Path,
    record_property,
) -> None:
    wav = fixture_dir / wav_name
    async with VoiceSession(ws_url) as session:
        await session.send_wav(wav)
        await session.wait_for_event("tts_first_audio_sample", timeout=15.0)
        ttft = session.compute_ttft_audio_ms()

    record_property("case_id", case_id)
    record_property("ttft_audio_ms", ttft)
    record_property("reference", reference)
    assert ttft < max_ttft_ms, f"{case_id}: TTFTAudio {ttft:.1f}ms >= {max_ttft_ms}ms"
