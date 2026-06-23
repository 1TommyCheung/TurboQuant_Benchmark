# voice-agent — Pattern C real-time voice loop

Real-time `ASR -> LLM -> TTS` voice agent built on the TurboQuant_Benchmark
stack. Targets **<800 ms speech-end-to-first-audio** on a split-GPU topology:
the LLM runs in Docker on an RTX 4090, the audio stack (Whisper + Kokoro +
Silero VAD) runs as a single Python process pinned to an RTX 3090 Ti. The
split is not isolation theater — it eliminates the 80-400 ms CUDA-serialization
TTS stutters that appear when the LLM is decoding while Kokoro is synthesizing
on the same device.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Host: WSL2 Ubuntu 24.04, CUDA 13.2                                 │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ RTX 4090 (24 GB, sm_89)                                 │       │
│  │   beellama-server (Docker)                              │       │
│  │     ghcr.io/1tommycheung/beellama-server:stable         │       │
│  │     Qwen3.5-9B Q4_K_M @ 4K, F16 KV                      │       │
│  │     port 8083, ~7 GB VRAM, 160 ms TTFT, 126 tok/s       │       │
│  └─────────────────────────────────────────────────────────┘       │
│                          │                                         │
│                          │ HTTP /v1/chat/completions (stream)      │
│                          ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ audio-service (Python, uvicorn)  CUDA_VISIBLE_DEVICES=1 │       │
│  │   faster-whisper large-v3-turbo INT8   ~1.6 GB          │       │
│  │   Kokoro-82M v1.0 KPipeline            ~1.0 GB          │       │
│  │   Silero VAD (CPU, onnxruntime)        ~5 MB            │       │
│  │   FastAPI WebSocket on :8090                            │       │
│  └─────────────────────────────────────────────────────────┘       │
│  RTX 3090 Ti: ~2.6 GB used, ~21 GB free                            │
└────────────────────────────────────────────────────────────────────┘
```

---

## Quick start

Prereqs:

| Requirement | Notes |
|---|---|
| Docker + Compose v2 | with NVIDIA Container Toolkit |
| Two NVIDIA GPUs | device 0 = 4090, device 1 = 3090 Ti (see `docker-compose.yml`) |
| Model file | `${HOME}/models/Qwen3.5-9B-Q4_K_M.gguf` mounted into the LLM container |
| WSL2 + CUDA 13.2 | host already configured for the rest of the benchmark suite |

```bash
cd voice-agent
cp .env.example .env            # optional; defaults work in-compose
docker compose up --build
```

Expected boot:

1. `voice-beellama` pulls the image, loads the GGUF, `/v1/models` healthcheck
   goes green (~60 s cold).
2. `voice-audio-service` builds, downloads Whisper + Kokoro weights into
   the `audio-cache` volume on first run (~1.5 GB), then opens port 8090.
3. `GET http://localhost:8090/healthz` returns `{"status":"ok"}`.

Tear down: `docker compose down`. Add `-v` to drop the model cache volume.

---

## VRAM expectations

| Component | GPU | Steady-state VRAM | Notes |
|---|---|---|---|
| beellama (Qwen3.5-9B Q4_K_M @ 4K, F16 KV) | RTX 4090 (dev 0) | **~7 GB** | leaves ~16 GB headroom |
| faster-whisper large-v3-turbo INT8 | RTX 3090 Ti (dev 1) | ~1.6 GB | |
| Kokoro-82M v1.0 | RTX 3090 Ti (dev 1) | ~1.0 GB | |
| Silero VAD | CPU | ~5 MB | onnxruntime |
| **Audio stack total** | RTX 3090 Ti | **~2.6 GB** | ~21 GB free for future workloads |

If `nvidia-smi` shows the audio process on device 0, double-check
`CUDA_VISIBLE_DEVICES=1` is set in the container — that's the most common
misconfiguration.

---

## Talking to the agent

The audio-service speaks a single WebSocket on `ws://localhost:8090/ws`:

| Direction | Frame | Format |
|---|---|---|
| client -> server | binary | 16 kHz mono PCM int16, ~32 ms (512 sample) chunks |
| server -> client | binary | 24 kHz mono PCM int16 TTS audio |
| server -> client | text (JSON) | event stream: `{"type":"vad_end"}`, `{"type":"stt","text":"..."}`, `{"type":"llm_token","text":"..."}`, `{"type":"turn_end"}` |

### wscat (smoke test)

```bash
# Health only — does not send audio.
curl -fsS http://localhost:8090/healthz

# Open the socket and watch the JSON event stream.
wscat -c ws://localhost:8090/ws
```

`wscat` is convenient for inspecting events but is not great at streaming raw
binary at a fixed cadence. Use the Python client for actual audio.

### Python client snippet

```python
import asyncio, json, wave
import numpy as np
import websockets

async def speak(wav_path: str) -> None:
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    async with websockets.connect("ws://localhost:8090/ws") as ws:
        # Stream in 32 ms chunks so VAD sees realistic timing.
        chunk = 512  # 32 ms @ 16 kHz
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i + chunk].tobytes())
            await asyncio.sleep(0.032)

        # Collect events + audio response until turn_end.
        async for frame in ws:
            if isinstance(frame, bytes):
                print(f"audio: {len(frame)} bytes")
            else:
                evt = json.loads(frame)
                print("event:", evt)
                if evt.get("type") == "turn_end":
                    break

asyncio.run(speak("bench/voice/audio/fixtures/medium_time.wav"))
```

### curl is not enough

There is no plain HTTP endpoint for audio in/out — voice is streaming both
ways, and curl's WebSocket support is too limited to send timed binary
frames. Use `wscat` for inspection or the Python client for real traffic.

---

## Running the benchmark

The harness lives in `bench/voice/` and is driven by pytest. It assumes the
compose stack is already up.

```bash
# from voice-agent/
pip install -r bench/voice/requirements.txt

# core latency suite (T01-T05)
pytest bench/voice/test_latency.py -v

# concurrency suite (1 / 2 / 5 sessions)
pytest bench/voice/test_concurrency.py -v

# chaos / fault-injection
pytest bench/voice/test_chaos.py -v

# write a JSON report and compare to baseline.json
python -m bench.voice.report --out runs/$(date +%Y-%m-%dT%H%M).json
```

Pass criteria (warm path):

| ID | Input | Target |
|---|---|---|
| T01_short_greeting | `short_hi.wav` (0.4 s) | TTFTAudio < 600 ms |
| T02_medium_factual | `medium_time.wav` (1.2 s) | TTFTAudio < 800 ms |
| T03_long_explanation | `long_explain.wav` (3.0 s) | TTFTAudio < 1200 ms |
| T04_barge_in | barge_in_pair/ | original stream aborted within 200 ms |
| T05_concurrent_5 | 5x T02 | p95 TTFTAudio < 1600 ms |

`TTFTAudio = tts_first_audio_sample - audio_chunk_sent_last` (monotonic).

---

## Further reading

- **Full design + benchmark plan:** [`docs/voice-agent/2026-06-23-pattern-c-design.md`](../docs/voice-agent/2026-06-23-pattern-c-design.md)
  — synthesizes 8 parallel agent investigations: topology decision, latency
  budget, component specs, hardening items, benchmark harness, open questions.
- **Smoke checklist:** [`SMOKE.md`](./SMOKE.md)
- Component upstreams: [hexgrad/kokoro](https://github.com/hexgrad/kokoro),
  [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2),
  [snakers4/silero-vad](https://github.com/snakers4/silero-vad),
  [beellama-server](https://github.com/1TommyCheung/beellama.cpp/pkgs/container/beellama-server).

---

## Known limitations

- **Two-GPU requirement.** Single-GPU operation is unsupported. With both
  models on one device, CUDA serialization produces audible 80-400 ms TTS
  stutters whenever the LLM is mid-decode. See design doc §"Resolved tension".
- **WSL2 only.** Paths, NVIDIA toolkit flags, and audio loopback timings are
  tuned for WSL2 + CUDA 13.2. Bare-metal Linux should work but is untested.
- **English only.** Whisper is loaded with `language="en"` and Kokoro uses
  `lang_code='a'` (American English) + the `af_heart` voice.
- **No turn-detector second pass.** Endpointing is Silero VAD with
  `min_silence_duration_ms=450`. This is the single biggest latency knob and
  can mis-cut "um... actually" pauses; LiveKit turn-detector is on the
  follow-up list.
- **Full-utterance Whisper, not streaming.** STT fires on VAD endpoint, not
  incrementally during speech. Streaming would save ~100-150 ms; deferred
  until baseline is established.
- **Barge-in cancels but doesn't refund.** When the user interrupts, in-flight
  LLM generation is aborted and pending TTS audio is dropped via the per-turn
  UUID guard — but tokens already produced are wasted compute.
- **No auth on the WebSocket.** Anyone with network access to port 8090 can
  open a session. Localhost-only by default; do not expose without a proxy.
- **No browser client shipped.** The audio output format choice (24 kHz raw
  PCM int16 vs Opus) for in-browser playback is still open; see design doc
  §"Open questions".
- **Models live on host.** `${HOME}/models/Qwen3.5-9B-Q4_K_M.gguf` must
  exist on ext4 (not `/mnt/` NTFS) for acceptable load times. Whisper and
  Kokoro weights are downloaded into the `audio-cache` Docker volume on
  first run.
- **No priority scheduling yet.** beellama `max_num_seqs=2` + voice-loop
  `priority=10` is recommended hardening in the design doc but not wired up;
  expect TTFT collapse if you point background batch traffic at port 8083
  while a voice session is live.
