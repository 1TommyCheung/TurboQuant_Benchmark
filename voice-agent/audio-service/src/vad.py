"""Silero VAD wrapper for streaming voice activity detection.

Streaming pattern
-----------------
Client audio arrives over a WebSocket as fixed-size float32 PCM frames at
16 kHz (``window_size_samples`` = 512 → 32 ms per chunk). Each chunk is fed
into a Silero ``VADIterator`` which is *stateful*: it integrates probability
over time and emits ``{"start": <sample_idx>}`` once speech begins and
``{"end": <sample_idx>}`` once ``min_silence_duration_ms`` of trailing
silence has elapsed.

This wrapper translates those raw transitions into our orchestrator's event
schema and, crucially, accumulates the per-utterance audio between
``start`` and ``end`` so the STT stage receives a single contiguous buffer
with ``speech_pad_ms`` of context on either side already baked in by
``VADIterator``.

A small pre-roll ring buffer keeps the most recent chunks so that, when a
``start`` fires, the leading audio (which was already streamed past) is
prepended to the utterance — Silero's own ``speech_pad_ms`` only handles
the trailing side cleanly.

Lifecycle:
  * one :class:`SileroVAD` instance per WebSocket connection
  * :meth:`process_chunk` is awaited per incoming frame
  * :meth:`reset_states` is called on disconnect (or to discard a turn)
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Deque, Literal, Optional, TypedDict

import numpy as np
from silero_vad import VADIterator, load_silero_vad

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .config import Settings


class VADEvent(TypedDict, total=False):
    """Event emitted by :meth:`SileroVAD.process_chunk`."""

    event: Literal["start", "end"]
    audio: np.ndarray  # float32 PCM @ 16kHz; present on both 'start' and 'end'


class SileroVAD:
    """Async-friendly wrapper around Silero ``VADIterator``.

    The underlying ONNX model is CPU-only and tiny (~5 MB). Inference per
    32 ms chunk is well under 1 ms on a modern CPU core, so we run it
    inline on the event loop's default executor to avoid blocking the
    WebSocket coroutine.
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._sampling_rate: int = int(settings.SAMPLE_RATE_IN)
        self._window_size: int = int(settings.VAD_WINDOW_SAMPLES)

        # CPU-only via onnxruntime
        self._model = load_silero_vad(onnx=True)
        self._iterator: VADIterator = self._build_iterator()

        # Accumulator for the current utterance (between start/end).
        self._utterance: list[np.ndarray] = []
        self._in_speech: bool = False

        # Pre-roll ring buffer of recent chunks so a 'start' event can
        # include the audio leading into the trigger. Sized to roughly
        # ``speech_pad_ms`` worth of chunks.
        pad_chunks = max(
            1,
            int(
                (settings.VAD_SPEECH_PAD_MS / 1000.0)
                * self._sampling_rate
                / self._window_size
            ),
        )
        self._preroll: Deque[np.ndarray] = deque(maxlen=pad_chunks)

        # Input buffer: accumulate arbitrary-size incoming chunks and feed the
        # Silero model in fixed window_size windows. Real WebSocket/mic streams
        # send variable chunk sizes; without this, any sub-window chunk raises
        # "Input audio chunk is too short".
        self._inbuf: list[np.ndarray] = []
        self._inbuf_len: int = 0

        # Speculative silence-onset detection (energy-based, independent of
        # Silero's neural endpoint). Emits one 'silence_onset' per speech run.
        self._spec_enabled: bool = bool(
            getattr(settings, "STT_SPECULATIVE", True)
        )
        self._spec_silence_rms: float = float(
            getattr(settings, "STT_SPEC_SILENCE_RMS", 0.012)
        )
        self._spec_silence_windows: int = int(
            getattr(settings, "STT_SPEC_SILENCE_WINDOWS", 5)
        )
        self._spec_armed: bool = False
        self._spec_silence_run: int = 0

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _build_iterator(self) -> VADIterator:
        return VADIterator(
            self._model,
            threshold=float(self._settings.VAD_THRESHOLD),
            sampling_rate=self._sampling_rate,
            min_silence_duration_ms=int(self._settings.VAD_MIN_SILENCE_MS),
            speech_pad_ms=int(self._settings.VAD_SPEECH_PAD_MS),
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    async def process_chunk(
        self, audio_chunk: np.ndarray
    ) -> Optional[VADEvent]:
        """Feed one window of PCM audio to the VAD.

        Parameters
        ----------
        audio_chunk:
            ``float32`` numpy array of shape ``(window_size_samples,)`` at
            16 kHz. Values are expected in ``[-1.0, 1.0]``.

        Returns
        -------
        ``None`` if no transition occurred, otherwise a dict with:
          * ``event``: ``"start"`` or ``"end"``
          * ``audio``: leading pre-roll on start, full utterance on end
        """

        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32, copy=False)

        # Buffer incoming audio and process in fixed window_size windows.
        # Real WebSocket/mic streams send variable chunk sizes; Silero
        # requires exactly window_size samples per inference call.
        self._inbuf.append(audio_chunk)
        self._inbuf_len += int(audio_chunk.size)

        event: Optional[dict] = None
        while self._inbuf_len >= self._window_size:
            combined = np.concatenate(self._inbuf)
            window = combined[: self._window_size]
            remainder = combined[self._window_size :]
            self._inbuf = [remainder] if remainder.size else []
            self._inbuf_len = int(remainder.size)

            window_event = await self._process_window(window)
            if window_event is not None and event is None:
                # Return the first event in this batch; subsequent events
                # in the same input chunk are rare (speech transitions
                # don't happen within ~32ms) and ignored.
                event = window_event

        return event

    async def _process_window(self, window: np.ndarray) -> Optional[dict]:
        """Run Silero on one fixed-size window and update tracking state."""
        loop = asyncio.get_running_loop()
        raw: Optional[dict] = await loop.run_in_executor(
            None, self._iterator, window
        )

        # Always feed the utterance buffer when we're inside speech, and
        # keep the pre-roll ring up to date when we're not.
        if self._in_speech:
            self._utterance.append(window)
        else:
            self._preroll.append(window)

        # Speculative silence-onset detection. While inside speech, watch the
        # raw RMS energy. The first window that drops to "silence" emits a
        # lightweight 'silence_onset' event carrying the utterance-so-far, so
        # the orchestrator can speculatively start STT during Silero's
        # MIN_SILENCE confirmation window. Silero still owns the real 'end'.
        if raw is None:
            if (
                self._spec_enabled
                and self._in_speech
                and not self._spec_armed
            ):
                rms = float(np.sqrt(np.mean(window.astype(np.float32) ** 2)))
                if rms < self._spec_silence_rms:
                    self._spec_silence_run += 1
                else:
                    # Voiced window — reset; this was an intra-word pause.
                    self._spec_silence_run = 0
                # Only arm after a sustained run of silence: a single quiet
                # window is just a pause between words and would truncate the
                # utterance mid-sentence (Whisper then hallucinates).
                if self._spec_silence_run >= self._spec_silence_windows:
                    self._spec_armed = True
                    utt = (
                        np.concatenate(self._utterance)
                        if self._utterance
                        else np.zeros(0, dtype=np.float32)
                    )
                    return {"event": "silence_onset", "audio": utt}
            return None

        if "start" in raw:
            self._in_speech = True
            self._spec_armed = False  # new speech run: re-arm speculation
            self._spec_silence_run = 0
            # Seed the utterance buffer with the pre-roll so STT receives
            # the audio leading into the trigger point.
            preroll = list(self._preroll)
            self._preroll.clear()
            self._utterance = list(preroll)
            # Include the current chunk if it wasn't already appended.
            if not preroll or preroll[-1] is not window:
                self._utterance.append(window)
            leading = (
                np.concatenate(self._utterance)
                if self._utterance
                else np.zeros(0, dtype=np.float32)
            )
            return {"event": "start", "audio": leading}

        if "end" in raw:
            self._in_speech = False
            self._spec_armed = False
            self._spec_silence_run = 0
            utterance = (
                np.concatenate(self._utterance)
                if self._utterance
                else np.zeros(0, dtype=np.float32)
            )
            self._utterance = []
            return {"event": "end", "audio": utterance}

        return None

    def reset_states(self) -> None:
        """Discard all state. Call on WebSocket disconnect / turn abort."""

        try:
            self._iterator.reset_states()
        except AttributeError:  # pragma: no cover - older silero builds
            self._iterator = self._build_iterator()
        self._utterance = []
        self._preroll.clear()
        self._in_speech = False
        self._inbuf = []
        self._inbuf_len = 0
        self._spec_armed = False
        self._spec_silence_run = 0

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def in_speech(self) -> bool:
        """Whether the VAD is currently inside a speech segment."""

        return self._in_speech

    @property
    def window_size_samples(self) -> int:
        """Expected chunk size in samples (e.g. 512 → 32 ms at 16 kHz)."""

        return self._window_size
