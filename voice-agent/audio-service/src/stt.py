"""faster-whisper STT wrapper with subprocess isolation and auto-recycle.

ctranslate2 (the backend of faster-whisper) leaks roughly 2 MB per
``transcribe`` call. Over a long-running voice session this can accumulate
to multiple GB of resident memory in the audio service process. To bound
that growth we run the actual model in a *child process* and recycle that
child every ``settings.STT_RECYCLE_EVERY`` calls (default 500), reclaiming
all leaked memory wholesale.

Architecture
------------

    +-- parent (asyncio) --------------------------------+
    |  WhisperPool                                       |
    |    - multiprocessing.Pipe (audio bytes <-> text)   |
    |    - asyncio.Lock (serialize transcribe calls)     |
    |    - counter; on >= recycle_every: respawn child   |
    +----------------------------+-----------------------+
                                 | Pipe (duplex, pickle)
    +----------------------------v-----------------------+
    |  child: _whisper_worker()                          |
    |    - loads WhisperModel once                       |
    |    - loop: recv(audio_np) -> transcribe -> send    |
    |    - exits on sentinel None                        |
    +----------------------------------------------------+

The blocking ``Connection.recv`` in the parent is run via
``asyncio.to_thread`` so the event loop stays responsive. Calls are
serialized through an ``asyncio.Lock`` because a single child cannot
service overlapping requests.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Child process worker
# ---------------------------------------------------------------------------

def _whisper_worker(
    conn: Connection,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    cpu_threads: int,
    num_workers: int,
) -> None:
    """Run in a child process. Load Whisper once, then service requests.

    Wire protocol on ``conn``:
      parent -> child: ``np.ndarray`` (float32 mono 16kHz) or ``None`` to quit
      child -> parent: ``str`` transcript, or ``("__error__", repr(exc))``
    """
    # Import inside the child so heavy CUDA init does not happen in the parent.
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
            cpu_threads=cpu_threads,
        )
    except Exception as exc:  # noqa: BLE001
        conn.send(("__error__", f"model_load_failed: {exc!r}"))
        conn.close()
        return

    # Signal readiness.
    conn.send("__ready__")

    while True:
        try:
            payload = conn.recv()
        except EOFError:
            break

        if payload is None:
            # Graceful shutdown sentinel.
            break

        audio: np.ndarray = payload
        try:
            segments, _info = model.transcribe(
                audio,
                language=language,
                beam_size=1,            # greedy = fastest
                vad_filter=False,       # upstream Silero handles VAD
                without_timestamps=True,
            )
            text = "".join(seg.text for seg in segments)
            conn.send(text)
        except Exception as exc:  # noqa: BLE001
            conn.send(("__error__", repr(exc)))

    conn.close()


# ---------------------------------------------------------------------------
# Parent-side pool
# ---------------------------------------------------------------------------

class WhisperPool:
    """Async-friendly pool that owns exactly one Whisper child process.

    The child is transparently recycled every
    ``settings.STT_RECYCLE_EVERY`` calls to bound ctranslate2's slow
    memory leak (~2 MB/call).
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._model_name: str = getattr(
            settings, "STT_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2"
        )
        self._device: str = getattr(settings, "STT_DEVICE", "cuda")
        self._compute_type: str = getattr(settings, "STT_COMPUTE_TYPE", "int8")
        self._language: str = getattr(settings, "STT_LANGUAGE", "en")
        self._cpu_threads: int = getattr(settings, "STT_CPU_THREADS", 4)
        self._num_workers: int = getattr(settings, "STT_NUM_WORKERS", 1)
        self._recycle_every: int = getattr(settings, "STT_RECYCLE_EVERY", 500)

        # "spawn" is required for CUDA-using children (fork + CUDA is UB).
        self._mp_ctx = mp.get_context("spawn")

        self._proc: Optional[mp.process.BaseProcess] = None
        self._conn: Optional[Connection] = None
        self._lock = asyncio.Lock()
        self._calls: int = 0
        self._shutdown: bool = False

        self._spawn_child()

    # ------------------------------------------------------------------
    # Child lifecycle
    # ------------------------------------------------------------------

    def _spawn_child(self) -> None:
        parent_conn, child_conn = self._mp_ctx.Pipe(duplex=True)
        proc = self._mp_ctx.Process(
            target=_whisper_worker,
            args=(
                child_conn,
                self._model_name,
                self._device,
                self._compute_type,
                self._language,
                self._cpu_threads,
                self._num_workers,
            ),
            name="whisper-worker",
            daemon=True,
        )
        proc.start()
        # Parent does not need the child's end of the pipe.
        child_conn.close()

        # Wait for readiness handshake (blocking, but only at construction/recycle).
        ready = parent_conn.recv()
        if isinstance(ready, tuple) and ready and ready[0] == "__error__":
            proc.terminate()
            raise RuntimeError(f"Whisper child failed to start: {ready[1]}")
        if ready != "__ready__":
            proc.terminate()
            raise RuntimeError(f"Whisper child sent unexpected handshake: {ready!r}")

        self._proc = proc
        self._conn = parent_conn
        self._calls = 0
        logger.info(
            "WhisperPool: spawned child pid=%s model=%s device=%s compute=%s",
            proc.pid,
            self._model_name,
            self._device,
            self._compute_type,
        )

    def _kill_child(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)  # graceful sentinel
            except (BrokenPipeError, OSError):
                pass
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

        proc = self._proc
        self._proc = None
        if proc is None:
            return

        proc.join(timeout=2.0)
        if proc.is_alive():
            logger.warning(
                "WhisperPool: child pid=%s did not exit; terminating", proc.pid
            )
            proc.terminate()
            proc.join(timeout=2.0)
        if proc.is_alive():
            logger.error(
                "WhisperPool: child pid=%s still alive after terminate; killing",
                proc.pid,
            )
            proc.kill()
            proc.join(timeout=1.0)

    def _recycle(self) -> None:
        logger.info(
            "WhisperPool: recycling child after %d calls (limit=%d)",
            self._calls,
            self._recycle_every,
        )
        self._kill_child()
        self._spawn_child()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 16kHz audio array to text.

        Serializes calls through an internal lock (one child = one
        in-flight request). Recycles the child if the call count has
        reached ``STT_RECYCLE_EVERY``.
        """
        if self._shutdown:
            raise RuntimeError("WhisperPool is shut down")

        # faster-whisper expects float32 mono at 16 kHz.
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        async with self._lock:
            if self._calls >= self._recycle_every:
                await asyncio.to_thread(self._recycle)

            if (
                self._conn is None
                or self._proc is None
                or not self._proc.is_alive()
            ):
                logger.warning("WhisperPool: child not alive; respawning")
                await asyncio.to_thread(self._recycle)

            assert self._conn is not None  # for type checker

            def _roundtrip() -> str:
                assert self._conn is not None
                self._conn.send(audio)
                resp = self._conn.recv()
                if isinstance(resp, tuple) and resp and resp[0] == "__error__":
                    raise RuntimeError(f"Whisper child error: {resp[1]}")
                if not isinstance(resp, str):
                    raise RuntimeError(
                        f"Whisper child sent unexpected payload: {type(resp).__name__}"
                    )
                return resp

            try:
                text = await asyncio.to_thread(_roundtrip)
            except (EOFError, BrokenPipeError, OSError) as exc:
                logger.error("WhisperPool: pipe broken (%s); recycling", exc)
                await asyncio.to_thread(self._recycle)
                # One retry after recycle.
                text = await asyncio.to_thread(_roundtrip)

            self._calls += 1
            return text

    async def shutdown(self) -> None:
        """Tear down the child process. Idempotent."""
        if self._shutdown:
            return
        self._shutdown = True
        async with self._lock:
            await asyncio.to_thread(self._kill_child)
        logger.info("WhisperPool: shutdown complete")
