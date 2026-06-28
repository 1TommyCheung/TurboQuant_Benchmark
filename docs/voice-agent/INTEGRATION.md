# Voice Agent Pipeline — Integration Guide

Portable reference for embedding the **Pattern C real-time voice pipeline**
(ASR → LLM → TTS) into another project. Self-contained: a consuming project
needs only Docker + two NVIDIA GPUs (or one) and a WebSocket client.

- **Source:** `voice-agent/` in this repo (`1TommyCheung/TurboQuant_Benchmark`)
- **Design rationale:** `docs/voice-agent/2026-06-23-pattern-c-design.md`
- **Validated:** 2026-06-24, RTX 4090 + RTX 3090 Ti, WSL2, CUDA 13.2

---

## 1. What this is

A self-hosted, OpenAI-free voice agent: speak into a WebSocket, get streamed
audio back. No cloud STT/TTS/LLM — everything runs on local GPUs.

```
                      ┌──────────────── audio-service (one container) ─────────────────┐
  mic (16kHz PCM) ──▶ │  Silero VAD (CPU)  →  faster-whisper STT  →  ┐                  │
                      │                                              │ HTTP stream      │
                      │                                              ▼                  │
                      │                              beellama LLM ◀──┘ (separate        │
                      │                              (OpenAI-compat)    container/GPU)  │
                      │                                              │                  │
  audio (24kHz PCM) ◀─│  Kokoro TTS  ◀───── _strip_think ◀──────────┘                  │
                      └───────────────────────────────────────────────────────────────┘
```

**Measured warm latency (speech-end → first audio): ~570-660ms** across short
and long turns. Under the 800ms conversational target. See §8.

---

## 2. Components & why each was chosen

| Role | Component | Where it runs | VRAM | Why |
|---|---|---|---|---|
| **LLM** | beellama Qwen3.5-9B Q4_K_M @ 4K, F16 KV | GPU 0 (4090) | ~7 GB | 140ms TTFT, 126 tok/s; published image `ghcr.io/1tommycheung/beellama-server:stable` |
| **STT** | faster-whisper `large-v3-turbo` INT8 | GPU 1 (3090 Ti) | ~1.6 GB | <100ms on short utterances; subprocess-isolated for leak control |
| **TTS** | Kokoro-82M v1.0 | GPU 1 | ~1 GB | ~140ms first-chunk, 24kHz, Apache-2.0 |
| **VAD** | Silero VAD (ONNX) | CPU | 0 | <1ms/chunk; endpoint authority |

Alternatives considered and rejected are documented in the design doc
(Whisper variants, Piper/Coqui/XTTS for TTS, etc.).

---

## 3. The WebSocket contract (the integration surface)

This is the **only interface** a consuming project needs. Everything else is
internal.

**Endpoint:** `ws://<host>:8090/ws/voice` (one session per connection)

| Direction | Format |
|---|---|
| **Client → server** | binary frames, **16 kHz mono float32 PCM** (little-endian) |
| **Server → client** | binary frames, **24 kHz mono float32 PCM** |

**Protocol rules:**
1. Stream mic audio as it's captured — any frame size works (the VAD buffers
   internally to 512-sample windows). ~20-40ms frames are typical.
2. **Keep the connection open and keep streaming** (including silence) for the
   whole conversation. The server's VAD detects speech/endpoints from the
   continuous stream. Do NOT close after one utterance.
3. Server pushes 24kHz audio frames as the assistant speaks. Play them as they
   arrive (streamed sentence-by-sentence).
4. **Barge-in:** just keep streaming the mic. If the user speaks while the
   assistant is talking, the server detects it, cancels the in-flight
   response, and starts listening — no client action needed.

**Health check:** `GET http://<host>:8090/health` → 200 when the service is up.
(Note: returns 200 *before* models finish loading — for true readiness, open a
WS and send a short utterance, or poll until first response works.)

### Minimal client (Python)

```python
import asyncio, wave, numpy as np, websockets

async def talk(wav_path, ws_url="ws://localhost:8090/ws/voice"):
    w = wave.open(wav_path, "rb")  # 16kHz mono s16
    pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    chunk = int(16000 * 0.032)  # 32ms frames
    frames = [pcm[i:i+chunk] for i in range(0, len(pcm), chunk)]
    silence = np.zeros(chunk, np.float32)
    resp = bytearray()

    async with websockets.connect(ws_url, max_size=None) as ws:
        async def send():
            for f in frames:
                await ws.send(f.astype(np.float32).tobytes()); await asyncio.sleep(0.032)
            while True:  # keep mic open with silence
                await ws.send(silence.tobytes()); await asyncio.sleep(0.032)
        async def recv():
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    resp.extend(msg); return  # got first audio
        s = asyncio.create_task(send())
        await recv(); s.cancel()
    return np.frombuffer(bytes(resp), np.float32)  # 24kHz mono
```

### Browser client (Web Audio API)

- Capture mic with `getUserMedia` + `AudioWorklet`, downsample to 16kHz,
  send `Float32Array.buffer` frames over the WebSocket.
- Receive frames, feed into an `AudioBufferSourceNode` queue at 24kHz.
- Raw float32 PCM is sent on the wire (no Opus) — simplest path; add Opus
  later if bandwidth matters.

---

## 4. Deploy in another project (the fast path)

The pipeline is two containers wired by one `docker-compose.yml`. Copy the
`voice-agent/` directory into the target project, or reference it.

```bash
# 1. Copy the directory (or git submodule it)
cp -r voice-agent/ /path/to/your-project/

# 2. Provide the LLM model on an ext4 path (NOT NTFS/WSL /mnt — 3-5x slower load)
mkdir -p ~/models
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models/

# 3. Configure
cd /path/to/your-project/voice-agent
cp .env.example .env        # edit if needed (defaults are production-tuned)

# 4. Build the audio-service image + pull the LLM image, start both
docker compose up -d --build

# 5. Wait for readiness, then connect a client to ws://localhost:8090/ws/voice
```

The `beellama` image is pulled pre-built from GHCR; only `audio-service` builds
locally (~8-12 min first time — CUDA 12.6 base + ~30 pip packages, then cached).

---

## 5. GPU topology options

| Topology | LLM | Audio stack | When |
|---|---|---|---|
| **Split (default)** | GPU 0 | GPU 1 | **Recommended** — avoids CUDA serialization stutter when LLM decodes while TTS synthesizes (they overlap in streaming). Crash isolation; ~39 GB free across both cards. |
| **Single GPU** | GPU 0 | GPU 0 | Works if you only have one card. ~13 GB total. Risk: occasional 80-400ms TTS audio gaps under GPU contention. Set both services to `device_ids: ['0']`. |

Pin via `docker-compose.yml` `device_ids` + the `CUDA_VISIBLE_DEVICES` env on
audio-service. Silero VAD is CPU-only either way.

**Note on base images:** the LLM container is CUDA **13** (beellama was compiled
against it); the audio-service container is CUDA **12.6** (faster-whisper's
ctranslate2 + torch + onnxruntime ship CUDA-12 wheels needing `libcublas.so.12`).
The host's CUDA 13 driver is backward-compatible with both. Don't "unify" them.

---

## 6. Configuration (env vars)

All in `.env` (read by docker-compose). Defaults are production-tuned; the ones
you're most likely to touch:

| Var | Default | Effect |
|---|---|---|
| `LLM_BASE_URL` | `http://beellama:8083/v1` | OpenAI-compatible LLM endpoint (compose DNS) |
| `LLM_SYSTEM_PROMPT` | "concise voice assistant… begin with 'Sure,'…" | **The leading-comma instruction matters** — TTS fires on the first clause (see §7) |
| `VAD_MIN_SILENCE_MS` | `300` | Endpoint silence wait. Lower = snappier, higher = fewer mid-pause cutoffs. Dominant latency knob. |
| `VAD_THRESHOLD` | `0.45` | Silero speech sensitivity (0.4-0.5 sweet spot) |
| `STT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Swap to `Systran/faster-distil-whisper-large-v3` or a `small.en` for faster/lighter |
| `STT_SPECULATIVE` | `true` | Hide STT latency under the VAD silence wait (see §7) |
| `STT_SPEC_SILENCE_WINDOWS` | `5` | Consecutive silent 32ms windows before speculating. **Lower = riskier** (truncates utterances → Whisper hallucinates). Keep ≥4. |
| `TTS_VOICE` | `af_heart` | Kokoro voice (54 options across 8 langs) |
| `CONTEXT_SIZE` | `4096` | LLM context — small for low TTFT |

---

## 7. The two latency optimizations (why it's fast)

Both are in `audio-service/src/orchestrator.py` and `vad.py`. Understand these
before tuning.

### Clause-boundary TTS firing (`_tts_from_tokens`)
The LLM is prompted to open replies with "Sure," / "Well,". The **first** TTS
chunk of each turn fires on a clause boundary (`, ; :` as well as `.!?`), so
synthesis starts on that 1-word acknowledgement (~140ms of audio) instead of
waiting for the first full sentence (~25 words). Subsequent chunks revert to
sentence boundaries for natural prosody. **Saved ~340ms.**

### Speculative STT (`vad.py` silence-onset → `orchestrator._spec_stt_start`)
STT normally runs *after* the VAD confirms end-of-speech. Instead, when the VAD
sees `STT_SPEC_SILENCE_WINDOWS` consecutive silent windows (160ms), it emits a
`silence_onset` event and the orchestrator starts transcribing the
utterance-so-far **in parallel** with Silero's remaining MIN_SILENCE
confirmation. If no speech resumes, the transcript is already done → STT latency
hidden. If speech resumes, the speculative result is discarded and a fresh
transcribe runs. **Silero remains the sole endpoint authority** — zero quality
risk. **Hides ~250ms on long turns.**

> ⚠️ The 5-window gate is a quality guard. A single low-energy window is just an
> intra-word pause; speculating on it truncates the utterance and Whisper
> hallucinates (observed: "explain how neural networks…" → "Thank you."). Don't
> drop `STT_SPEC_SILENCE_WINDOWS` below ~4.

### `_strip_think`
Qwen3.5 emits `<think></think>` tags even with `enable_thinking=false`; Kokoro's
phonemizer chokes on them. The orchestrator strips them from the token stream
before TTS. Keep this if you swap to any reasoning-capable LLM.

---

## 8. Latency breakdown (measured, warm)

Server-side `[timing]` instrumentation (in `orchestrator.py`), espeak synthetic
audio, localhost, single-user, n=2-3:

| Stage | Time | Notes |
|---|---|---|
| VAD silence wait | 300ms | dead air before endpoint commit |
| STT | ~10ms | hidden by speculation (was 250-360ms) |
| LLM TTFT | ~140ms | beellama Q4_K_M |
| TTS first chunk | ~140ms | Kokoro, clause-fire |
| **speech-end → first audio** | **~570-660ms** | under 800ms target |

**Measurement caveats (read before trusting these for production):**
- Synthetic espeak audio is cleaner than real speech — real-mic STT/VAD slower
- localhost = zero network latency; a remote client adds 10-100ms
- Single-turn, single-user; no concurrency/conversation-growth tested
- "Accuracy" was eyeballed transcript match, not formal WER
- The `bench/voice/` pytest harness exists but needs the keep-alive client fix
  + real audio fixtures for defensible p50/p95 numbers

---

## 9. File map (what to read when modifying)

```
voice-agent/
├── docker-compose.yml              # 2-service wiring, GPU pinning, network
├── .env.example                    # all config knobs
├── docker/Dockerfile.audio-service # CUDA 12.6 base, deps, spaCy model preinstall
├── audio-service/src/
│   ├── main.py                     # FastAPI app, /ws/voice, /health, lifecycle
│   ├── orchestrator.py             # THE pipeline: VAD→STT→LLM→TTS, barge-in,
│   │                               #   speculative STT, clause-fire, _strip_think
│   ├── vad.py                      # Silero wrapper, input buffering, silence-onset
│   ├── stt.py                      # faster-whisper subprocess pool + recycle
│   ├── tts.py                      # Kokoro KPipeline, sentence streaming
│   ├── turn_controller.py          # barge-in generation_id UUID guard
│   └── config.py                   # pydantic-settings (env surface)
└── docs/HARDENING.md               # production failure modes + mitigations
```

**Most edits land in `orchestrator.py`** — it's the conductor. The component
wrappers (vad/stt/tts) are deliberately thin and swappable.

---

## 10. Swapping components

The pipeline is built around clean interfaces so pieces can be replaced:

| Swap | How |
|---|---|
| **Different LLM** | Point `LLM_BASE_URL` at any OpenAI-compatible endpoint (vLLM, Ollama, etc.). Keep `_strip_think` if it emits reasoning tags. |
| **Different STT** | Change `STT_MODEL` (any faster-whisper CT2 model), or replace `stt.py`'s `WhisperPool` (interface: `async transcribe(np.ndarray) -> str`). |
| **Different TTS** | Replace `tts.py`'s `KokoroTTS` (interface: `synthesize(text, voice) -> AsyncIterator[np.ndarray]` at 24kHz). Piper/XTTS/Edge-TTS all fit. |
| **Different VAD** | Replace `vad.py`'s `SileroVAD` (interface: `async process_chunk(np.ndarray) -> Optional[dict]` emitting start/silence_onset/end). |

---

## 11. Known gaps / production TODO

From `docs/HARDENING.md`:
- Whisper subprocess recycle every 500 calls (ctranslate2 leak) — implemented
- Barge-in `generation_id` guard — implemented
- VAD `reset_states` on disconnect — implemented
- **Not yet:** beellama `max_num_seqs`/priority for multi-user; LiveKit
  turn-detector for smarter endpointing; Opus audio transport; formal WER bench;
  load test ≥5 concurrent sessions.

For a real product, wire up the `bench/voice/` harness with CommonVoice audio
and `jiwer` WER scoring before trusting the latency numbers in §8.
