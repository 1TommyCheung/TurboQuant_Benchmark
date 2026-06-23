"""Turn controller — barge-in safety via generation_id UUIDs.

Real-time voice agents must handle *barge-in*: the user starts speaking
while the agent is still talking. When that happens, every in-flight
artifact tied to the old turn (LLM tokens, TTS audio chunks, queued
playback buffers) must be discarded the instant the user's voice is
detected — otherwise the agent keeps talking over the user.

This module centralises that invariant in a single source of truth: the
*current turn ID*. Every producer (LLM streaming, TTS synthesiser) and
every consumer (audio player) stamps or checks work against the current
turn ID. The moment a new turn is started, all prior IDs become stale
and any chunk tagged with a stale ID is dropped on the floor.

Barge-in flow
-------------
1. VAD detects ``on_speech_start`` while the agent is mid-utterance.
2. Orchestrator calls ``await controller.cancel_current()``.
   - This atomically rotates ``current_id`` to a fresh UUID.
   - Every previously-issued ID is now stale (``is_active`` -> False).
3. Orchestrator calls ``sd.stop()`` to halt the sounddevice output stream.
4. Orchestrator calls ``llm_client.abort_generation()`` if available.
5. Orchestrator clears pending token / audio queues.
6. STT begins capturing the user's new utterance.
7. When STT is done, orchestrator calls ``await controller.new_turn()``
   to stamp the response cycle and dispatches to the LLM.

Producers must stamp every emitted chunk with the turn ID active at the
time the chunk was *produced*. Consumers must call ``is_active(turn_id)``
before acting on the chunk; if it returns False, drop the chunk.

The async lock around ``new_turn`` / ``cancel_current`` prevents two
concurrent barge-ins (e.g. VAD glitch + orchestrator retry) from racing
to install different IDs.
"""

from __future__ import annotations

import asyncio
import uuid


class TurnController:
    """Single source of truth for the active turn ID.

    Async safety: ``new_turn`` and ``cancel_current`` are serialised
    through an ``asyncio.Lock``. ``current_id`` and ``is_active`` are
    lock-free reads — attribute assignment is atomic in CPython, and a
    stale read simply means a chunk gets dropped one cycle later, which
    is harmless.
    """

    def __init__(self) -> None:
        self._current_id: uuid.UUID = uuid.uuid4()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def new_turn(self) -> uuid.UUID:
        """Atomically generate a fresh turn ID, invalidating prior turns.

        Returns the new ID. Callers should stamp every chunk they emit
        for this turn with this value.
        """
        async with self._lock:
            self._current_id = uuid.uuid4()
            return self._current_id

    def current_id(self) -> uuid.UUID:
        """Return the currently active turn ID (lock-free read)."""
        return self._current_id

    def is_active(self, turn_id: uuid.UUID) -> bool:
        """True iff ``turn_id`` matches the currently active turn.

        Producers and consumers call this on every chunk; a False result
        means the chunk belongs to a turn that has since been cancelled
        or superseded, and must be discarded.
        """
        return turn_id == self._current_id

    async def cancel_current(self) -> None:
        """Invalidate the current turn by rotating to a fresh ID.

        This is the barge-in entry point. After this returns, every
        chunk tagged with the previously-active ID will fail
        ``is_active`` and be dropped.

        The caller is also responsible for:
          - ``sd.stop()`` to halt audio playback immediately
          - ``llm_client.abort_generation()`` if the backend supports it
          - clearing any pending token / audio queues

        These side effects live in the orchestrator rather than here so
        the controller stays a pure ID arbiter with no I/O dependencies.
        """
        async with self._lock:
            self._current_id = uuid.uuid4()
