"""Kokoro KPipeline wrapper for text-to-speech.

Streaming pattern
-----------------
The voice agent's LLM produces tokens one-at-a-time. We can't wait for the
full response before speaking — that would blow our TTFA budget. Instead we
buffer incoming tokens into a rolling text buffer and *flush* on sentence
boundaries (period / exclamation / question mark followed by whitespace).

For each completed sentence (>= MIN_SENTENCE_CHARS), we hand the chunk to
Kokoro's KPipeline, which yields 24 kHz mono float32 audio. We re-yield
those audio chunks downstream (to the WebRTC sender or playback writer).

Any text left in the buffer when the token stream closes is flushed as a
final synthesis call so we don't drop the tail of the utterance.

Barge-in safety
---------------
The user can interrupt the agent mid-sentence ("barge-in"). When that
happens, the TurnController invalidates the current `generation_id`. Before
yielding *each* audio chunk we re-check `turn_controller.is_active(gen_id)`
and silently drop the chunk if the generation is stale. This guarantees we
never push audio for a turn the user has already interrupted, even if
KPipeline is mid-flight on a long sentence.

The TTFA-critical path is: token in -> sentence boundary -> KPipeline ->
first audio chunk out. Keep sentences short on the LLM side for best
latency.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

if TYPE_CHECKING:
    from .config import Settings
    from .turn_controller import TurnController


# Sentence boundary: . ! or ? followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Don't synthesize chunks shorter than this — too-short fragments produce
# awkward prosody and waste a KPipeline call.
MIN_SENTENCE_CHARS = 10

# Kokoro pipeline output sample rate.
SAMPLE_RATE = 24_000


class KokoroTTS:
    """Async wrapper around `kokoro.KPipeline` with sentence-boundary streaming."""

    def __init__(self, settings: "Settings") -> None:
        # Imported lazily so unit tests can stub the module without a GPU.
        from kokoro import KPipeline  # type: ignore[import-not-found]

        self._settings = settings
        self._voice: str = getattr(settings, "TTS_VOICE", "af_heart")
        self._pipeline = KPipeline(lang_code="a")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    async def stream_tokens_to_audio(
        self,
        token_stream: AsyncIterator[str],
        generation_id: uuid.UUID,
        turn_controller: "TurnController",
    ) -> AsyncIterator[np.ndarray]:
        """Buffer LLM tokens, flush on sentence boundaries, yield 24kHz float32 audio.

        Before yielding each audio chunk we check
        ``turn_controller.is_active(generation_id)`` — if the turn was
        interrupted (barge-in), the chunk is discarded.
        """
        buffer = ""

        async for token in token_stream:
            # Cheap staleness check on the input side too, so we stop
            # buffering work for a turn the user has already killed.
            if not turn_controller.is_active(generation_id):
                return

            if not token:
                continue
            buffer += token

            # Flush every complete sentence we can find in the buffer.
            while True:
                match = _SENTENCE_BOUNDARY.search(buffer)
                if match is None:
                    break

                sentence = buffer[: match.end()].strip()
                buffer = buffer[match.end() :]

                if len(sentence) < MIN_SENTENCE_CHARS:
                    # Too short — stitch it onto the next sentence by
                    # putting it back at the front of the buffer.
                    buffer = sentence + " " + buffer
                    break

                async for audio in self._synthesize(sentence):
                    if not turn_controller.is_active(generation_id):
                        return
                    yield audio

        # Drain any tail text the LLM left in the buffer.
        tail = buffer.strip()
        if tail:
            async for audio in self._synthesize(tail):
                if not turn_controller.is_active(generation_id):
                    return
                yield audio

    async def synth_single(self, text: str) -> AsyncIterator[np.ndarray]:
        """One-shot synthesis for tests / non-streaming fallback paths."""
        text = text.strip()
        if not text:
            return
        async for audio in self._synthesize(text):
            yield audio

    async def synthesize(
        self, text: str, voice: str | None = None
    ) -> AsyncIterator[np.ndarray]:
        """Public adapter used by the orchestrator.

        Accepts a per-call voice override (falls back to the instance default).
        Yields 24 kHz float32 mono PCM chunks.
        """
        text = (text or "").strip()
        if not text:
            return
        async for audio in self._synthesize(text, voice=voice):
            yield audio

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    async def _synthesize(
        self, text: str, voice: str | None = None
    ) -> AsyncIterator[np.ndarray]:
        """Run KPipeline off the event loop and yield float32 mono audio chunks."""
        loop = asyncio.get_running_loop()
        # KPipeline returns a generator of (graphemes, phonemes, audio).
        # We materialise it in a worker thread so we don't block the loop,
        # but we yield each chunk back to the loop as it arrives.
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        def _produce() -> None:
            try:
                for _gs, _ps, audio in self._pipeline(
                    text, voice=voice or self._voice
                ):
                    arr = _to_float32_mono(audio)
                    loop.call_soon_threadsafe(queue.put_nowait, arr)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        producer = loop.run_in_executor(None, _produce)

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            await producer


def _to_float32_mono(audio: object) -> np.ndarray:
    """Coerce KPipeline output (torch.Tensor or ndarray) to float32 mono ndarray."""
    # KPipeline typically returns a torch.Tensor; avoid a hard torch import.
    arr_obj = audio
    to_numpy = getattr(arr_obj, "detach", None)
    if callable(to_numpy):
        arr_obj = arr_obj.detach().cpu().numpy()  # type: ignore[union-attr]

    arr = np.asarray(arr_obj)
    if arr.ndim > 1:
        # Mix down to mono if Kokoro ever returns multi-channel output.
        arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr
