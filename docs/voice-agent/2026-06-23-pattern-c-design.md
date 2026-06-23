# Voice Agent Pattern C — Synthesized Design + Benchmark Plan

**Synthesized from 8 parallel subagent investigations, 2026-06-23**
**Hardware:** RTX 4090 24 GB (sm_89) + RTX 3090 Ti 24 GB (sm_86), WSL2, CUDA 13.2
**Goal:** Real-time voice loop (ASR → LLM → TTS) targeting **<800ms speech-end-to-first-audio**

---

## TL;DR

| Decision | Choice | Rationale |
|---|---|---|
| **GPU topology** | **Split: LLM on 4090, audio on 3090 Ti** ⭐ | CUDA serialization on a single GPU causes 80-400ms TTS audio stutters when LLM streams tokens concurrently with TTS synthesis. Split eliminates this. |
| **LLM** | beellama Qwen3.5-9B Q4_K_M @ 4K, F16 KV | 160ms warm TTFT, 126 tok/s decode, 7 GB VRAM, validated |
| **STT** | `faster-whisper` `large-v3-turbo` @ INT8 | ~1.6 GB VRAM, <100ms transcribe on 2-3s utterance |
| **TTS** | Kokoro 82M v1.0 via `KPipeline` | ~1 GB VRAM, 150ms first-audio chunk, 24kHz mono PCM |
| **VAD** | Silero VAD on **CPU** | <1ms per chunk, zero GPU footprint, 450ms silence threshold |
| **Architecture** | Pattern C — LLM in Docker, audio stack as single Python process | Crash isolation for LLM, low-overhead audio path |
| **Realistic latency** | **~430-580ms warm** (well under 800ms target) | Theoretical floor ~280-360ms |

---

## Resolved tension: single 4090 vs dual-GPU split

Three subagents weighed in on topology with conflicting recommendations:

- **#2 (single 4090):** "Pipeline is sequential, no parallelism to exploit, splitting gains nothing."
- **#3 (split LLM/audio):** "Crash isolation, headroom on 3090 Ti for future workloads."
- **#8 (failure modes):** "CUDA serialization on a single GPU causes 80-400ms audible TTS stutters when LLM is mid-decode while Kokoro requests synthesis."

**Tiebreaker = #8.** The streaming sentence-by-sentence pattern means **LLM IS decoding while TTS is synthesizing** — they ARE concurrent in steady state. On a single GPU, CUDA serializes these by default, producing audible audio gaps. **Split topology is the correct call** not for isolation theater but for actual audio integrity in a real-time loop.

---

## Stage-by-stage latency budget (synthesized from agent #1)

| Stage | Cold latency | Warm latency | Cumulative (warm) |
|---|---|---|---|
| VAD endpoint detection | 50ms | 30ms | 30ms |
| Audio transport (WSL2 loopback) | 10ms | 5ms | 35ms |
| Whisper STT (streaming residual) | 150ms | 50ms | 85ms |
| **LLM TTFT (beellama)** | 405ms | **160ms** | **245ms** |
| LLM first-sentence decode (~15 tokens) | 120ms | 80ms | 325ms |
| TTS first-chunk synthesis (Kokoro) | 250ms | 150ms | 475ms |
| Audio buffer + playback start | 30ms | 25ms | 500ms |
| **Total** | **~1015ms** | **~500ms** | |

**Biggest tuning knob:** `min_silence_duration_ms` in VAD (default 450ms, contributes more than any other single stage). Dropping to 300ms saves 150ms but risks cutting off "um... actually" pauses. Recommend starting at 450ms.

**Second biggest:** streaming Whisper during user speech (saves ~100-150ms vs waiting for endpoint then batching).

---

## Final architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Host: WSL2 Ubuntu 24.04, CUDA 13.2                                 │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ RTX 4090 (24 GB, sm_89)                                  │      │
│  │  ┌────────────────────────────────────────────────────┐  │      │
│  │  │ beellama-server (Docker container)                 │  │      │
│  │  │   ghcr.io/1tommycheung/beellama-server:stable      │  │      │
│  │  │   Qwen3.5-9B-Q4_K_M @ 4K, F16 KV                   │  │      │
│  │  │   port 8083, ~7 GB VRAM                            │  │      │
│  │  │   CUDA_VISIBLE_DEVICES=0                           │  │      │
│  │  └────────────────────────────────────────────────────┘  │      │
│  │  Headroom: ~16 GB (room for reranker / draft model)      │      │
│  └──────────────────────────────────────────────────────────┘      │
│                            │                                        │
│                            │ HTTP /v1/chat/completions (streaming)  │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ audio-service (Python uvicorn process)                   │      │
│  │   CUDA_VISIBLE_DEVICES=1  (pinned to 3090 Ti)            │      │
│  │   ┌─────────────────────┐ ┌───────────────────────────┐ │      │
│  │   │ faster-whisper      │ │ Kokoro KPipeline          │ │      │
│  │   │ large-v3-turbo INT8 │ │ Kokoro-82M v1.0           │ │      │
│  │   │ ~1.6 GB on 3090 Ti  │ │ ~1.0 GB on 3090 Ti        │ │      │
│  │   └─────────────────────┘ └───────────────────────────┘ │      │
│  │   ┌─────────────────────────────────────────────────┐    │      │
│  │   │ Silero VAD (CPU only, onnxruntime)              │    │      │
│  │   │ ~5 MB, threshold=0.45, min_silence=450ms        │    │      │
│  │   └─────────────────────────────────────────────────┘    │      │
│  │   FastAPI WebSocket on port 8090 (client audio in/out)   │      │
│  └──────────────────────────────────────────────────────────┘      │
│  RTX 3090 Ti VRAM total: ~2.6 GB used, ~21 GB free                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## Component specs (consolidated from subagents 4, 5, 6)

### LLM (validated)

```yaml
image: ghcr.io/1tommycheung/beellama-server:stable
env:
  MODEL_PATH: /models/Qwen3.5-9B-Q4_K_M.gguf
  CACHE_TYPE_K: f16
  CACHE_TYPE_V: f16
  CONTEXT_SIZE: 4096
  REASONING_FORMAT: none
gpus: device=0  # 4090
```

For hardening: append `max_num_seqs=2` + priority scheduling to prevent TTFT collapse under concurrent users.

### STT (Whisper) — from subagent #4

```python
from faster_whisper import WhisperModel

stt = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cuda",
    compute_type="int8",   # ~1.6 GB VRAM
    num_workers=1,
    cpu_threads=4,
)

def transcribe(audio_np: np.ndarray) -> str:
    segments, _ = stt.transcribe(
        audio_np, language="en",
        beam_size=1,           # greedy = fastest
        vad_filter=False,      # upstream Silero handles
        without_timestamps=True,
    )
    return "".join(s.text for s in segments)
```

### TTS (Kokoro) — from subagent #5

```python
from kokoro import KPipeline
import re

tts = KPipeline(lang_code='a')  # American English
VOICE = 'af_heart'
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

async def llm_tokens_to_audio(token_stream):
    buf = ""
    async for token in token_stream:
        buf += token
        parts = SENTENCE_END.split(buf)
        while len(parts) > 1:
            sentence = parts.pop(0).strip()
            buf = " ".join(parts)
            if len(sentence) >= 10:
                for audio_chunk, sr in tts(sentence, voice=VOICE):
                    yield audio_chunk  # 24kHz float32
    if buf.strip():
        for audio_chunk, sr in tts(buf.strip(), voice=VOICE):
            yield audio_chunk
```

**Latency trick (from #5):** prepend a system instruction like *"Always start replies with a brief acknowledgment like 'Sure,' or 'Well,'"* — gives TTS a comma-terminated fragment to start synthesizing immediately, saving 100-200ms of perceived responsiveness.

### VAD (Silero) — from subagent #6

```python
from silero_vad import load_silero_vad, VADIterator

vad_model = load_silero_vad(onnx=True)  # CPU-only via onnxruntime
vad = VADIterator(
    vad_model,
    threshold=0.45,                      # 0.4-0.5 sweet spot
    sampling_rate=16000,
    min_silence_duration_ms=450,         # KEY latency knob
    speech_pad_ms=100,                   # tail padding for utterance boundaries
)
# window_size_samples=512 → 32ms chunks
```

Run in a dedicated thread with a ring-buffer of raw audio chunks. Fire `on_speech_start` for barge-in detection (interrupt TTS), `on_speech_end` for STT trigger.

---

## Top 5 day-one hardening items (from subagent #8)

### 1. Barge-in stale audio guard

```python
# Per-turn UUID; TTS consumer checks before writing each chunk
class TurnController:
    def __init__(self):
        self.current_turn_id = uuid.uuid4()
        self.lock = asyncio.Lock()

    async def new_turn(self):
        async with self.lock:
            self.current_turn_id = uuid.uuid4()
            return self.current_turn_id

    def is_active(self, turn_id):
        return turn_id == self.current_turn_id

# TTS playback loop
async def play(audio_chunk, my_turn_id):
    if not turn_ctrl.is_active(my_turn_id):
        return  # discard stale chunk
    await audio_out.write(audio_chunk)

# On barge-in (Silero on_speech_start)
async def on_barge_in():
    sd.stop()                          # halt audio output IMMEDIATELY
    await turn_ctrl.new_turn()         # invalidates pending chunks
    await llm_client.abort_generation()  # stop wasting LLM compute
    await llm_token_queue.clear()
```

### 2. Kokoro on 3090 Ti (CUDA stream isolation) ✓ already in our split topology

### 3. Whisper subprocess + scheduled recycle

```python
# Wrap Whisper in a multiprocessing subprocess with auto-recycle
class WhisperPool:
    def __init__(self, recycle_every=500):
        self.proc = self._spawn()
        self.calls = 0
        self.recycle_every = recycle_every

    def transcribe(self, audio):
        self.calls += 1
        if self.calls >= self.recycle_every:
            self._restart()
        return self.proc.transcribe(audio)
```

### 4. VAD state reset on WebSocket disconnect

```python
@websocket.on_disconnect
async def on_disconnect(ws):
    vad.reset_states()  # Silero ONNX model has this
    audio_buffer.clear()
    await turn_ctrl.new_turn()  # invalidate any pending TTS
```

### 5. beellama priority scheduling

Add `priority=10` (high) on voice-loop requests, `priority=1` (low) on background tasks. Cap `max_num_seqs=2` to prevent TTFT collapse under load.

---

## Benchmark harness (from subagent #7)

### File structure

```
bench/voice/
├── conftest.py              # shared fixtures: ws client, audio clock
├── audio/
│   ├── fixtures/            # pre-recorded WAVs (16kHz mono PCM)
│   │   ├── short_hi.wav             # "hi" — 0.4s
│   │   ├── medium_time.wav          # "what time is it" — 1.2s
│   │   ├── long_explain.wav         # "explain how X works" — 3.0s
│   │   └── barge_in_pair/           # interrupter + interrupted
│   └── capture.py           # AudioCapture: WS stream → numpy + timestamps
├── harness.py               # VoiceSession: ingest WAV → stream → record events
├── metrics.py               # compute TTFTAudio, WER, turn_duration
├── test_latency.py          # core latency cases (T01-T05)
├── test_concurrency.py      # 1/2/5 concurrent sessions
├── test_chaos.py            # fault injection
└── report.py                # JSON + regression vs baseline.json
```

### Primary metric

**TTFTAudio** = `tts_first_audio_sample.timestamp − audio_chunk_sent_last.timestamp`

Capture monotonic event timestamps: `audio_chunk_sent_last`, `vad_end_detected`, `stt_result_received`, `llm_first_token`, `tts_first_audio_sample`.

### Top 5 test cases

| ID | Input | Pass criteria |
|----|---|---|
| **T01_short_greeting** | "hi" (0.4s WAV) | TTFTAudio < 600ms |
| **T02_medium_factual** | "what time is it" (1.2s) | TTFTAudio < 800ms |
| **T03_long_explanation** | "explain how neural networks work" (3s) | TTFTAudio < 1200ms, no timeout |
| **T04_barge_in** | 2s utterance, interrupt at 1s with new audio | original stream aborted within 200ms |
| **T05_concurrent_5** | 5 simultaneous T02 sessions | p95 TTFTAudio < 1600ms (≤2× solo) |

### Output format

```json
{
  "run_id": "2026-06-23T18:00:00",
  "baseline_id": "2026-06-23T14:00:00",
  "topology": "split_4090_llm_3090ti_audio",
  "cases": [
    {"id": "T01_short_greeting", "ttft_audio_ms": 541, "wer": 0.0, "tokens": 4},
    {"id": "T02_medium_factual", "ttft_audio_ms": 691, "wer": 0.0, "tokens": 12}
  ],
  "summary": {"p50_ttft_ms": 612, "p95_ttft_ms": 891, "max_concurrent": 5},
  "regression": {"T02_medium_factual": "+87ms vs baseline — WARN"}
}
```

### Chaos tests (`test_chaos.py`)

- Kill beellama mid-stream → client receives `{"error": "upstream_disconnected"}` within 2s, no partial audio play
- Drop WebSocket packet → graceful continuation
- Force VRAM near cap → degrade quality not crash
- Restart Kokoro process → next turn succeeds

---

## Implementation phases

| Phase | Goal | Effort |
|---|---|---|
| **P1** | Stand up `audio-service` Python skeleton on 3090 Ti with Whisper + Kokoro + Silero | 1 day |
| **P2** | Orchestrator (FastAPI WebSocket) tying VAD → STT → LLM HTTP → TTS streaming | 1 day |
| **P3** | Barge-in (`generation_id` guard + `sd.stop()`) | 0.5 day |
| **P4** | Benchmark harness `bench/voice/` with T01-T05 + chaos tests | 1 day |
| **P5** | First benchmark run, regression baseline | 0.5 day |
| **P6** | Hardening: Whisper subprocess recycle, VAD reset, priority scheduling | 1 day |
| | **Total** | **~5 days** |

---

## Open questions / next investigations

1. **Streaming Whisper vs full-utterance** — subagent #4 recommends full-utterance on VAD endpoint for simplicity; subagent #1 suggests streaming would save 100-150ms. Worth a follow-up A/B once baseline is up.
2. **LiveKit turn-detector as second-pass** — subagent #6 mentions; could reduce false endpoints by ~80%, at cost of ~50ms CPU. Try after baseline.
3. **System prompt "always start with a filler"** — subagent #5's perceptual-latency hack. Easy to A/B in T01-T03.
4. **MPS daemon for the LLM ↔ embedding model coexistence on 4090** — subagent #2 skipped it; would let us add a small reranker/embedding on the 4090 without context contention. Only relevant if we end up co-locating an additional model.
5. **Audio output format** for browser clients — 24kHz raw PCM int16 vs Opus encoding. Web Audio API supports raw; Opus saves bandwidth but adds encode/decode latency.

---

## Sources

Synthesized from 8 parallel investigations on 2026-06-23. Full per-agent reports in transcripts. Key external references:

- [hexgrad/kokoro](https://github.com/hexgrad/kokoro) — Kokoro TTS
- [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) — STT model
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) — VAD
- [livekit/agents turn-detector](https://github.com/livekit/agents) — semantic turn detection
- [beellama-server image](https://github.com/1TommyCheung/beellama.cpp/pkgs/container/beellama-server) — LLM container
