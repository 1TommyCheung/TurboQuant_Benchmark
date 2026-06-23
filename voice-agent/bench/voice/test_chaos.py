"""Chaos / fault-injection tests.

Cases:
    T04_barge_in              -> barge-in aborts original TTS stream within 200ms
    kill_beellama_mid_stream  -> client sees upstream_disconnected within 2s
    drop_ws_packet            -> graceful continuation
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from harness import VoiceSession


BARGE_IN_ABORT_BUDGET_MS = 200.0
LLM_UPSTREAM_TIMEOUT_S = 2.0


@pytest.mark.asyncio
@pytest.mark.wav_fixture("barge_in_pair/utterance_a.wav")
@pytest.mark.wav_fixture("barge_in_pair/utterance_b.wav")
async def test_T04_barge_in(ws_url: str, fixture_dir: Path, record_property) -> None:
    """Start a long turn, interrupt with a new utterance. The original TTS
    stream must be aborted within 200ms of the barge-in trigger."""
    a = fixture_dir / "barge_in_pair" / "utterance_a.wav"
    b = fixture_dir / "barge_in_pair" / "utterance_b.wav"

    async with VoiceSession(ws_url) as session:
        send_task = asyncio.create_task(session.send_wav(a))
        await session.wait_for_event("tts_first_audio_sample", timeout=15.0)

        # barge in: send a new utterance immediately
        t_barge = time.monotonic()
        session._events.append({"event": "barge_in_sent", "t": t_barge, "meta": {}})
        try:
            await session.send_wav(b)
        finally:
            send_task.cancel()
            try:
                await send_task
            except (asyncio.CancelledError, Exception):
                pass

        # server must emit a stream_aborted (or turn_invalidated) event
        try:
            ev = await session.wait_for_event("stream_aborted", timeout=2.0)
        except TimeoutError:
            ev = await session.wait_for_event("turn_invalidated", timeout=2.0)
        abort_latency_ms = (ev["t"] - t_barge) * 1000.0

    record_property("barge_in_abort_ms", abort_latency_ms)
    assert abort_latency_ms < BARGE_IN_ABORT_BUDGET_MS, (
        f"barge-in abort {abort_latency_ms:.1f}ms >= {BARGE_IN_ABORT_BUDGET_MS}ms budget"
    )


@pytest.mark.asyncio
@pytest.mark.wav_fixture("medium_time.wav")
async def test_kill_beellama_mid_stream(
    ws_url: str, fixture_dir: Path, record_property
) -> None:
    """Kill the LLM upstream while a turn is in flight; expect a clean error
    surfaced to the client within ``LLM_UPSTREAM_TIMEOUT_S``."""
    container = os.environ.get("BEELLAMA_CONTAINER", "beellama-server")
    if subprocess.run(["which", "docker"], capture_output=True).returncode != 0:
        pytest.skip("docker not available")

    wav = fixture_dir / "medium_time.wav"
    async with VoiceSession(ws_url) as session:
        send_task = asyncio.create_task(session.send_wav(wav))
        await session.wait_for_event("llm_first_token", timeout=10.0)

        subprocess.run(["docker", "kill", container], check=False, capture_output=True)
        t_kill = time.monotonic()

        try:
            ev = await session.wait_for_event("error", timeout=LLM_UPSTREAM_TIMEOUT_S + 1.0)
        finally:
            send_task.cancel()
            try:
                await send_task
            except (asyncio.CancelledError, Exception):
                pass

        latency = ev["t"] - t_kill
        record_property("upstream_error_latency_s", latency)
        assert latency < LLM_UPSTREAM_TIMEOUT_S, (
            f"upstream error surfaced after {latency:.2f}s >= {LLM_UPSTREAM_TIMEOUT_S}s"
        )
        err_msg = (ev["meta"].get("error", "") or "").lower()
        assert "upstream" in err_msg or "disconnected" in err_msg


@pytest.mark.asyncio
@pytest.mark.wav_fixture("medium_time.wav")
async def test_ws_drop_graceful(ws_url: str, fixture_dir: Path) -> None:
    """Closing the WebSocket mid-stream must not crash the audio-service; a
    reconnect immediately afterwards must succeed and produce audio."""
    wav = fixture_dir / "medium_time.wav"

    # First session: drop mid-send
    session = VoiceSession(ws_url)
    await session.connect()
    send_task = asyncio.create_task(session.send_wav(wav))
    await asyncio.sleep(0.2)
    await session.close()
    send_task.cancel()
    try:
        await send_task
    except (asyncio.CancelledError, Exception):
        pass

    # Second session must work
    async with VoiceSession(ws_url) as s2:
        await s2.send_wav(wav)
        await s2.wait_for_event("tts_first_audio_sample", timeout=15.0)
        assert s2.compute_ttft_audio_ms() > 0
