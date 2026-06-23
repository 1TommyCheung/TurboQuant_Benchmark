# voice-agent — Smoke / Bring-up Notes

Pattern C real-time voice agent. The LLM (beellama Qwen3.5-9B Q4_K_M) runs as
its own Docker container on the RTX 4090; this repo houses the **audio-service**
(VAD + STT + TTS + orchestrator) which runs on the RTX 3090 Ti, plus a
benchmark harness.

## File structure

```
voice-agent/
├── docker-compose.yml          # beellama (LLM) + audio-service
├── .env.example                # copy to .env
├── docker/
│   └── Dockerfile.audio-service
├── audio-service/
│   ├── requirements.txt
│   └── src/
│       ├── __init__.py
│       ├── config.py            # pydantic-settings Settings
│       ├── main.py              # FastAPI app + /ws/voice WebSocket
│       ├── orchestrator.py      # VoicePipeline (VAD→STT→LLM→TTS)
│       ├── vad.py               # SileroVAD (async wrapper)
│       ├── stt.py               # WhisperPool (subprocess-isolated)
│       ├── tts.py               # KokoroTTS (24 kHz f32 chunks)
│       └── turn_controller.py   # TurnController (UUID-based barge-in)
└── bench/voice/
    ├── conftest.py              # pytest fixtures
    ├── harness.py               # VoiceSession (WebSocket client)
    ├── metrics.py               # TTFTAudio, WER, percentiles
    ├── report.py                # results.json writer
    ├── test_latency.py
    ├── test_concurrency.py
    ├── test_chaos.py
    └── audio/fixtures/          # WAV inputs (mostly placeholders in git)
```

## Static checks performed

- `python3 -m py_compile audio-service/src/*.py bench/voice/*.py` — **clean**.
- `python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"` — **valid**.

## Fixes applied during smoke

1. `main.py` imported `WhisperSTT` but `stt.py` exports `WhisperPool` — added an
   `as WhisperSTT` alias on the import.
2. `orchestrator._run_vad` was synchronous and read `event["type"]`, but
   `SileroVAD.process_chunk` is `async` and emits `event["event"]`. Made the
   helper async, `await`-ed the VAD result when it's a coroutine, and now read
   both `"event"` and `"type"` keys.
3. `orchestrator._tts_from_tokens` could infinite-loop on a sentence shorter
   than `_MIN_SENTENCE_CHARS` (re-attach → same split → same sentence). It
   now `break`s out of the inner split loop after the re-attach so the next
   token is required to make progress.
4. `KokoroTTS` only exposed `stream_tokens_to_audio` / `synth_single`, but the
   orchestrator calls `self.tts.synthesize(text, voice=...)`. Added a public
   `synthesize()` adapter that threads through to `_synthesize`, and added a
   `voice` override to `_synthesize` itself.
5. `Dockerfile.audio-service` `CMD` was `uvicorn main:app` but the module
   lives at `src/main.py`. Changed to `src.main:app`.

## Known limitations / what is NOT done here

- **No model downloads.** Whisper turbo (`deepdml/faster-whisper-large-v3-turbo-ct2`),
  Kokoro voices, and Silero VAD weights are pulled at first run by their
  libraries. The audio-service container has `HF_HOME=/app/.cache/huggingface`,
  but no warm cache is baked in.
- **No `pip install` was run** in this static pass. Imports like
  `kokoro`, `faster_whisper`, `silero_vad`, `onnxruntime` are reachable only
  at runtime, not at compile time (we guarded heavy imports with `TYPE_CHECKING`
  or moved them into the child process).
- **No GPU / audio device** was exercised. The audio-service container needs
  the host's RTX 3090 Ti via NVIDIA Container Toolkit. Browser/WebSocket clients
  supply the microphone — there is no host-side audio device wired up in this
  repo.
- **WAV fixtures under `bench/voice/audio/fixtures/`** are placeholders; tests
  marked `@pytest.mark.wav_fixture(...)` are auto-skipped when files are zero
  bytes (see `conftest.py`).

## What a developer must do to actually run it

1. **Build the LLM image** (already published):
   `docker pull ghcr.io/1tommycheung/beellama-server:stable`
2. **Place the GGUF**: ensure `~/models/Qwen3.5-9B-Q4_K_M.gguf` exists on the
   host (the compose file mounts `${HOME}/models:/models:ro`).
3. **Copy env**: `cp .env.example .env`, edit if needed.
4. **Build & start**:
   ```
   docker compose build audio-service
   docker compose up -d
   ```
5. **Wait for first-run model fetches** — Whisper-turbo (~1.6 GB) and Kokoro
   voices download into the `audio-cache` volume on first request. Watch
   `docker compose logs -f audio-service`.
6. **Sanity check**:
   ```
   curl -fsS http://localhost:8083/v1/models       # LLM
   curl -fsS http://localhost:8090/health          # audio-service
   ```
7. **End-to-end smoke** (any 16 kHz mono PCM source over WebSocket
   `ws://localhost:8090/ws/voice`):
   ```
   cd bench/voice
   pip install -r requirements.txt
   VOICE_WS_URL=ws://localhost:8090/ws/voice pytest -m wav_fixture
   ```
   (skips automatically if fixtures aren't populated).

## Quick smoke test sequence

```bash
# 1. Static
python3 -m py_compile audio-service/src/*.py bench/voice/*.py
python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"

# 2. Build
docker compose build audio-service

# 3. Run
docker compose up -d
docker compose ps                       # both healthy?
curl -fsS http://localhost:8083/v1/models | jq .
curl -fsS http://localhost:8090/health

# 4. Tail logs while you connect a WebRTC/WebSocket client
docker compose logs -f audio-service
```

## Health budget reminder (from design doc)

- TTFTAudio p95 ≤ 700 ms (target), ≤ 900 ms (ceiling).
- Barge-in abort latency < 200 ms.
- LLM upstream error surfacing < 2 s.
