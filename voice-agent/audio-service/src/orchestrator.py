"""Orchestrator wiring VAD, STT, LLM, and TTS into a streaming pipeline.

This module implements :class:`VoicePipeline`, the heart of Pattern C's real-time
voice loop. It consumes 16 kHz float32 PCM chunks coming off a WebSocket,
performs voice-activity detection, transcribes finished utterances with Whisper,
streams the transcript to the LLM over an OpenAI-compatible HTTP endpoint, and
synthesizes the reply sentence-by-sentence with Kokoro, fanning 24 kHz float32
audio chunks back out to a caller-supplied async callback.

Barge-in is handled via :class:`turn_controller.TurnController`: when the VAD
fires a ``start`` event while TTS is still streaming, the current turn id is
invalidated, the in-flight LLM generation is aborted, and any queued TTS chunks
are dropped before they can reach the WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
)

import numpy as np

logger = logging.getLogger(__name__)

# Split a buffer on sentence-ending punctuation followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Minimum characters before we flush a fragment to TTS. Avoids synthesizing
# tiny fragments like "Hi." (which would still work, but we want at least
# enough text for Kokoro to produce a smooth chunk).
_MIN_SENTENCE_CHARS = 10


class VoicePipeline:
    """End-to-end VAD -> STT -> LLM -> TTS pipeline.

    Parameters
    ----------
    settings:
        A pydantic settings object exposing ``LLM_MODEL``, ``LLM_SYSTEM_PROMPT``,
        ``LLM_BASE_URL`` (already wired into ``llm_client``), and a
        ``TTS_VOICE`` voice id.
    vad:
        A VAD wrapper exposing ``process_chunk(chunk) -> Optional[dict]`` where
        the returned dict has a ``type`` key in {"start", "end"} and (for end
        events) an ``audio`` key with the accumulated utterance as float32 PCM.
    stt:
        Speech-to-text wrapper exposing ``transcribe(audio: np.ndarray) -> str``
        (may be sync or async; both are supported).
    tts:
        Text-to-speech wrapper. ``tts.synthesize(text, voice) -> AsyncIterator``
        of 24 kHz float32 audio chunks.
    turn_controller:
        Manages per-turn identifiers for barge-in.
    llm_client:
        An ``openai.AsyncOpenAI`` instance pointing at the beellama server.
    """

    def __init__(
        self,
        settings: Any,
        vad: Any,
        stt: Any,
        tts: Any,
        turn_controller: Any,
        llm_client: Any,
    ) -> None:
        self.settings = settings
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self.turn_controller = turn_controller
        self.llm_client = llm_client

        # Conversation history (system prompt persists, user/assistant
        # turns appended as they happen). Kept short — Pattern C uses a
        # 4K context window.
        self._history: list[dict[str, str]] = [
            {
                "role": "system",
                "content": getattr(settings, "LLM_SYSTEM_PROMPT", ""),
            }
        ]

        # Active TTS task, so a barge-in can cancel it.
        self._tts_task: Optional[asyncio.Task[None]] = None
        # Handle to the current LLM stream (openai SDK) so we can close it.
        self._llm_stream: Any = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def handle_audio_stream(
        self,
        audio_in: AsyncIterator[np.ndarray],
        audio_out: Callable[[np.ndarray], Awaitable[None]],
    ) -> None:
        """Run the voice loop until ``audio_in`` is exhausted.

        Parameters
        ----------
        audio_in:
            Async iterator of 16 kHz float32 mono PCM chunks (already framed
            for Silero — typically 512 samples / 32 ms).
        audio_out:
            Async callable that accepts a 24 kHz float32 mono PCM chunk and
            forwards it to the client (WebSocket).
        """

        try:
            async for chunk in audio_in:
                event = await self._run_vad(chunk)
                if event is None:
                    continue

                etype = event.get("event") or event.get("type")
                if etype == "start":
                    await self._on_speech_start()
                elif etype == "end":
                    utterance = event.get("audio")
                    if utterance is None or len(utterance) == 0:
                        continue
                    await self._on_speech_end(utterance, audio_out)
        except asyncio.CancelledError:
            logger.info("handle_audio_stream cancelled")
            raise
        except Exception:  # noqa: BLE001
            logger.exception("voice loop crashed")
            raise
        finally:
            # Input stream ended. Drain any in-flight response so the client
            # receives the full audio before the WS closes. (In a live mic
            # session audio_in never exhausts; this matters for finite input
            # like test fixtures.) Barge-in still cancels immediately via
            # _on_speech_start -> _cancel_current_response.
            await self._drain_current_response()

    # ------------------------------------------------------------------ #
    # VAD event handlers
    # ------------------------------------------------------------------ #

    async def _run_vad(self, chunk: np.ndarray) -> Optional[dict[str, Any]]:
        """Feed one chunk to the VAD and normalize its return value.

        Supports both sync and async VAD wrappers (Silero wrapper is async).
        """
        result = self.vad.process_chunk(chunk)
        if asyncio.iscoroutine(result):
            result = await result
        if result is None:
            return None
        if isinstance(result, dict):
            return result
        # Some VAD wrappers may return a plain string ("start"/"end").
        return {"event": str(result)}

    async def _on_speech_start(self) -> None:
        """Barge-in: user started speaking while TTS may still be playing."""
        if self._tts_task is None or self._tts_task.done():
            return
        logger.info("barge-in detected, cancelling active turn")
        await self._cancel_current_response()

    async def _on_speech_end(
        self,
        utterance: np.ndarray,
        audio_out: Callable[[np.ndarray], Awaitable[None]],
    ) -> None:
        """User stopped speaking — transcribe, then kick off LLM+TTS."""
        transcript = await self._transcribe(utterance)
        transcript = (transcript or "").strip()
        if not transcript:
            logger.debug("empty transcript, skipping turn")
            return

        logger.info("user: %s", transcript)
        self._history.append({"role": "user", "content": transcript})

        turn_id = await self.turn_controller.new_turn()
        self._tts_task = asyncio.create_task(
            self._run_response(turn_id, audio_out)
        )

    # ------------------------------------------------------------------ #
    # Response pipeline (LLM -> TTS -> audio_out)
    # ------------------------------------------------------------------ #

    async def _run_response(
        self,
        turn_id: Any,
        audio_out: Callable[[np.ndarray], Awaitable[None]],
    ) -> None:
        """Stream one assistant turn: tokens -> sentences -> audio chunks."""
        assistant_text_parts: list[str] = []
        try:
            token_stream = self._strip_think(self._stream_llm_tokens(self._history))
            async for audio_chunk in self._tts_from_tokens(
                token_stream, assistant_text_parts
            ):
                if not self.turn_controller.is_active(turn_id):
                    logger.debug("turn %s superseded, dropping audio", turn_id)
                    return
                await audio_out(audio_chunk)
        except asyncio.CancelledError:
            logger.debug("response task cancelled (turn=%s)", turn_id)
            raise
        except Exception:  # noqa: BLE001
            logger.exception("response pipeline failed")
        finally:
            full = "".join(assistant_text_parts).strip()
            if full:
                self._history.append({"role": "assistant", "content": full})

    async def _stream_llm_tokens(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Yield tokens from the LLM as they arrive."""
        model = getattr(self.settings, "LLM_MODEL", "beellama")
        try:
            stream = await self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=getattr(self.settings, "LLM_TEMPERATURE", 0.7),
                max_tokens=getattr(self.settings, "LLM_MAX_TOKENS", 512),
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("failed to open LLM stream")
            return

        self._llm_stream = stream
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    yield token
        except asyncio.CancelledError:
            logger.debug("LLM stream cancelled")
            raise
        except Exception:  # noqa: BLE001
            logger.exception("LLM stream errored mid-flight")
        finally:
            await self._close_llm_stream()

    async def _strip_think(
        self, token_stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """Drop ``<think>...</think>`` blocks from the token stream.

        Qwen3.5 emits (often empty) think tags even with
        ``enable_thinking=false``; ``--reasoning-format none`` on beellama
        does not always strip them, and Kokoro's phonemizer cannot handle
        the raw tags — it logs "words count mismatch" and yields no audio.
        We buffer leading tokens until the closing ``</think>`` (or until
        it's clear there is no think block) and only then start emitting.
        """
        THRESHOLD = 256  # chars of buffered prefix before concluding no think block
        buffer = ""
        flushed = False
        async for token in token_stream:
            if flushed:
                yield token
                continue
            buffer += token
            if "</think>" in buffer:
                idx = buffer.rfind("</think>") + len("</think>")
                rest = buffer[idx:]
                flushed = True
                if rest.strip():
                    yield rest
            elif "<think>" not in buffer and len(buffer) >= THRESHOLD:
                # No think block at all — flush what we buffered.
                flushed = True
                yield buffer
        if not flushed:
            # Stream ended without a closing think tag; yield any residual
            # real content with tags stripped.
            cleaned = buffer.replace("<think>", "").replace("</think>", "")
            if cleaned.strip():
                yield cleaned

    async def _tts_from_tokens(
        self,
        token_stream: AsyncIterator[str],
        text_sink: list[str],
    ) -> AsyncIterator[np.ndarray]:
        """Group tokens into sentences, feed Kokoro, yield audio chunks."""
        voice = getattr(self.settings, "TTS_VOICE", "af_heart")
        buf = ""

        async for token in token_stream:
            text_sink.append(token)
            buf += token
            parts = _SENTENCE_END.split(buf)
            while len(parts) > 1:
                sentence = parts.pop(0).strip()
                buf = " ".join(parts)
                if len(sentence) >= _MIN_SENTENCE_CHARS:
                    async for audio_chunk in self._synthesize(sentence, voice):
                        yield audio_chunk
                else:
                    # Too short to bother — re-attach to the next fragment
                    # and stop splitting until more tokens arrive (else we
                    # would loop forever on the same boundary).
                    buf = f"{sentence} {buf}" if buf else sentence
                    break
                parts = _SENTENCE_END.split(buf)

        tail = buf.strip()
        if tail:
            async for audio_chunk in self._synthesize(tail, voice):
                yield audio_chunk

    async def _synthesize(
        self, text: str, voice: str
    ) -> AsyncIterator[np.ndarray]:
        """Adapter around the TTS wrapper (sync or async iterator)."""
        result = self.tts.synthesize(text, voice=voice)
        if hasattr(result, "__aiter__"):
            async for chunk in result:
                yield chunk
        else:
            # Sync iterable — yield chunks while letting the loop breathe.
            for chunk in result:
                yield chunk
                await asyncio.sleep(0)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _transcribe(self, audio: np.ndarray) -> str:
        """Call STT; supports both sync and async wrappers."""
        result = self.stt.transcribe(audio)
        if asyncio.iscoroutine(result):
            result = await result
        return result or ""

    async def _drain_current_response(self, timeout: float = 15.0) -> None:
        """Wait for the active response to finish sending audio.

        Called when the input stream ends so the client receives the full
        reply before the WebSocket closes. Falls back to a hard cancel if
        the response stalls past ``timeout``.
        """
        if self._tts_task is None or self._tts_task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._tts_task), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("response drain timed out after %.1fs, cancelling", timeout)
            await self._cancel_current_response()
        except Exception:  # noqa: BLE001
            logger.exception("response task errored during drain")

    async def _cancel_current_response(self) -> None:
        """Stop the active TTS task and abort the LLM stream."""
        await self.turn_controller.cancel_current()

        # Abort LLM if the client exposes a generic abort hook.
        abort = getattr(self.llm_client, "abort_generation", None)
        if callable(abort):
            try:
                res = abort()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001
                logger.exception("llm_client.abort_generation failed")

        await self._close_llm_stream()

        if self._tts_task is not None and not self._tts_task.done():
            self._tts_task.cancel()
            try:
                await self._tts_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tts_task = None

    async def _close_llm_stream(self) -> None:
        stream = self._llm_stream
        self._llm_stream = None
        if stream is None:
            return
        close = getattr(stream, "close", None) or getattr(
            stream, "aclose", None
        )
        if close is None:
            return
        try:
            res = close()
            if asyncio.iscoroutine(res):
                await res
        except Exception:  # noqa: BLE001
            logger.debug("LLM stream close errored", exc_info=True)
