"""FastAPI application entry point for the audio service.

This is the Pattern C audio-service process: it owns Silero VAD, faster-whisper
STT, Kokoro TTS, and the voice orchestrator. The LLM lives in a separate
Docker container (beellama on the RTX 4090); we talk to it over HTTP via the
``openai`` async client.

Run with::

    CUDA_VISIBLE_DEVICES=1 python -m audio_service.main

or via the bundled ``uvicorn.run`` guard at the bottom of this file.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from .config import Settings
from .orchestrator import VoicePipeline
from .stt import WhisperPool as WhisperSTT
from .tts import KokoroTTS
from .turn_controller import TurnController
from .vad import SileroVAD

logger = logging.getLogger(__name__)

app = FastAPI(title="voice-agent audio-service", version="0.1.0")


@app.on_event("startup")
async def _on_startup() -> None:
    """Instantiate every component and wire them into a :class:`VoicePipeline`."""
    settings = Settings()  # type: ignore[call-arg]
    logging.basicConfig(
        level=getattr(settings, "LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    vad = SileroVAD(settings)
    stt = WhisperSTT(settings)
    tts = KokoroTTS(settings)
    turn_controller = TurnController()

    llm_client = AsyncOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=getattr(settings, "LLM_API_KEY", "not-needed"),
    )

    pipeline = VoicePipeline(
        settings=settings,
        vad=vad,
        stt=stt,
        tts=tts,
        turn_controller=turn_controller,
        llm_client=llm_client,
    )

    app.state.settings = settings
    app.state.vad = vad
    app.state.stt = stt
    app.state.tts = tts
    app.state.turn_controller = turn_controller
    app.state.llm_client = llm_client
    app.state.pipeline = pipeline

    logger.info("audio-service ready (LLM=%s)", settings.LLM_BASE_URL)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Tear down subprocesses and HTTP clients."""
    stt = getattr(app.state, "stt", None)
    if stt is not None:
        try:
            shutdown = getattr(stt, "shutdown", None)
            if callable(shutdown):
                res = shutdown()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:  # noqa: BLE001
            logger.exception("STT shutdown failed")

    llm_client = getattr(app.state, "llm_client", None)
    if llm_client is not None:
        try:
            await llm_client.close()
        except Exception:  # noqa: BLE001
            logger.exception("LLM client close failed")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket) -> None:
    """Single voice session over WebSocket.

    Protocol:
      - inbound  : binary frames of 16 kHz float32 mono PCM
      - outbound : binary frames of 24 kHz float32 mono PCM
    """
    await websocket.accept()
    pipeline: VoicePipeline = app.state.pipeline
    turn_controller: TurnController = app.state.turn_controller
    vad: SileroVAD = app.state.vad

    disconnect = asyncio.Event()

    async def audio_in() -> AsyncIterator[np.ndarray]:
        while not disconnect.is_set():
            try:
                data = await websocket.receive_bytes()
            except WebSocketDisconnect:
                disconnect.set()
                return
            except Exception:  # noqa: BLE001
                logger.exception("websocket receive failed")
                disconnect.set()
                return
            if not data:
                continue
            chunk = np.frombuffer(data, dtype=np.float32)
            yield chunk

    async def audio_out(chunk: np.ndarray) -> None:
        if disconnect.is_set():
            return
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32, copy=False)
        try:
            await websocket.send_bytes(chunk.tobytes())
        except Exception:  # noqa: BLE001
            logger.exception("websocket send failed")
            disconnect.set()

    try:
        await pipeline.handle_audio_stream(audio_in(), audio_out)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("voice session crashed")
    finally:
        disconnect.set()
        # Reset per-session state so the next connection starts clean.
        try:
            reset = getattr(vad, "reset_states", None) or getattr(
                vad, "reset", None
            )
            if callable(reset):
                reset()
        except Exception:  # noqa: BLE001
            logger.debug("VAD reset failed", exc_info=True)
        try:
            await turn_controller.cancel_current()
        except Exception:  # noqa: BLE001
            logger.debug("turn cancel failed", exc_info=True)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def _main() -> None:
    """uvicorn entry point — used by both ``python -m`` and the script guard."""
    settings = Settings()  # type: ignore[call-arg]
    uvicorn.run(
        "audio_service.main:app"
        if __name__ != "__main__"
        else "src.main:app",
        host=getattr(settings, "HOST", "0.0.0.0"),
        port=getattr(settings, "PORT", 8090),
        log_level=getattr(settings, "LOG_LEVEL", "info").lower(),
        reload=False,
    )


if __name__ == "__main__":
    _main()
