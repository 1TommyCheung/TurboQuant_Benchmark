"""Voice benchmark harness.

VoiceSession connects to the audio-service WebSocket, streams a WAV file at
real-time rate (20ms chunks), captures audio out, and records monotonic event
timestamps for latency analysis.

Event log entries shape:
    {"event": str, "t": float, "meta": dict}

Standard event names (aligned with Pattern C design doc):
    audio_chunk_sent_first
    audio_chunk_sent_last
    vad_end_detected
    stt_result_received
    llm_first_token
    tts_first_audio_sample
    tts_last_audio_sample
"""

from __future__ import annotations

import asyncio
import json
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError:  # pragma: no cover - optional at import time
    websockets = None  # type: ignore[assignment]

try:
    import numpy as np
    import soundfile as sf
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    sf = None  # type: ignore[assignment]


# Audio constants for the 16kHz mono PCM client→server stream
SAMPLE_RATE = 16000
CHUNK_MS = 20
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000  # 320 samples
BYTES_PER_SAMPLE = 2  # int16
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE


class VoiceSession:
    """Connect to audio-service WebSocket, stream a WAV, capture audio out.

    Records monotonic event timestamps for latency analysis.
    """

    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self._ws: Any = None
        self._events: list[dict[str, Any]] = []
        self._audio_out: list[bytes] = []
        self._recv_task: asyncio.Task[None] | None = None
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> "VoiceSession":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets not installed")
        self._ws = await websockets.connect(self.ws_url, max_size=None)
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._record("ws_connected")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ events
    def _now(self) -> float:
        return time.monotonic()

    def _record(self, event: str, **meta: Any) -> None:
        self._events.append({"event": event, "t": self._now(), "meta": meta})

    def get_event_log(self) -> list[dict[str, Any]]:
        return list(self._events)

    # ------------------------------------------------------------------ send
    async def send_wav(self, wav_path: Path) -> None:
        """Stream a 16kHz mono WAV at real-time rate in 20ms chunks."""
        if sf is None or np is None:
            raise RuntimeError("numpy/soundfile not installed")

        wav_path = Path(wav_path)
        data, sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")
        if data.ndim > 1:
            data = data[:, 0]

        pcm = data.tobytes()
        first_sent = False
        start = self._now()
        for i, offset in enumerate(range(0, len(pcm), BYTES_PER_CHUNK)):
            chunk = pcm[offset : offset + BYTES_PER_CHUNK]
            if len(chunk) < BYTES_PER_CHUNK:
                # zero-pad final partial chunk to keep server-side framing simple
                chunk = chunk + b"\x00" * (BYTES_PER_CHUNK - len(chunk))
            await self._ws.send(chunk)
            if not first_sent:
                self._record("audio_chunk_sent_first")
                first_sent = True
            # real-time pacing
            target = start + (i + 1) * (CHUNK_MS / 1000.0)
            delay = target - self._now()
            if delay > 0:
                await asyncio.sleep(delay)
        self._record("audio_chunk_sent_last")

        # signal end-of-stream so the server can flush the VAD endpoint
        try:
            await self._ws.send(json.dumps({"type": "end_of_audio"}))
        except Exception:
            pass

    async def send_end_of_audio(self) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps({"type": "end_of_audio"}))

    # ------------------------------------------------------------------ receive
    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if isinstance(msg, (bytes, bytearray)):
                    if not self._audio_out:
                        self._record("tts_first_audio_sample", bytes=len(msg))
                    self._audio_out.append(bytes(msg))
                    self._events.append(
                        {"event": "tts_audio_chunk", "t": self._now(), "meta": {"bytes": len(msg)}}
                    )
                else:
                    try:
                        payload = json.loads(msg)
                    except Exception:
                        continue
                    ev = payload.get("event") or payload.get("type")
                    if ev:
                        self._record(ev, **{k: v for k, v in payload.items() if k not in ("event", "type")})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover
            self._record("ws_error", error=str(exc))

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Yield captured audio chunks as they arrive."""
        idx = 0
        while not self._closed:
            if idx < len(self._audio_out):
                yield self._audio_out[idx]
                idx += 1
            else:
                await asyncio.sleep(0.005)

    def audio_out_bytes(self) -> bytes:
        return b"".join(self._audio_out)

    # ------------------------------------------------------------------ metrics
    def event_t(self, name: str) -> float | None:
        for ev in self._events:
            if ev["event"] == name:
                return ev["t"]
        return None

    def compute_ttft_audio_ms(self) -> float:
        """TTFTAudio = tts_first_audio_sample - audio_chunk_sent_last (ms)."""
        t_last = self.event_t("audio_chunk_sent_last")
        t_first_audio = self.event_t("tts_first_audio_sample")
        if t_last is None or t_first_audio is None:
            raise RuntimeError(
                "missing events for TTFTAudio: "
                f"audio_chunk_sent_last={t_last}, tts_first_audio_sample={t_first_audio}"
            )
        return (t_first_audio - t_last) * 1000.0

    async def wait_for_event(self, name: str, timeout: float = 10.0) -> dict[str, Any]:
        deadline = self._now() + timeout
        while self._now() < deadline:
            for ev in self._events:
                if ev["event"] == name:
                    return ev
            await asyncio.sleep(0.01)
        raise TimeoutError(f"timed out waiting for event {name!r}")


def load_wav_duration_s(path: Path) -> float:
    """Return duration in seconds without loading entire payload into RAM."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())
