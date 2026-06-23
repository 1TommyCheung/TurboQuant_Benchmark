# Voice Agent — Production Hardening Checklist

Operational checklist for the Pattern C voice agent (split topology: RTX 4090 LLM / RTX 3090 Ti audio).
Source: [`docs/voice-agent/2026-06-23-pattern-c-design.md`](../../docs/voice-agent/2026-06-23-pattern-c-design.md), section "Top 5 day-one hardening items".

Each item below names a failure mode, where it lives in the code, the chaos test that exercises it, and the signal that tells you it broke in prod.

---

## Day-one hardening (P6 scope)

### 1. Barge-in stale audio guard (per-turn UUID)

- **What can go wrong:** After a barge-in, in-flight LLM tokens or queued TTS chunks from the previous turn replay over the user's new utterance, producing ghost speech and corrupting the dialogue state.
- **Where addressed:** `audio-service/src/turn_controller.py` — `TurnController` (holds `current_turn_id: UUID`, `new_turn()`, `is_active(turn_id)`). Consumed by the TTS write loop in `audio-service/src/tts.py` and by `audio-service/src/orchestrator.py` on `on_speech_start`.
- **How to verify (chaos test):** `bench/voice/test_chaos.py::test_barge_in_no_ghost_audio` — feed the `barge_in_pair/` fixture, assert the original TTS stream stops within 200ms and no PCM samples from turn N arrive after `new_turn()` returns turn N+1. Covered by **T04_barge_in** in `test_latency.py` for the timing budget.
- **Failure indicator:**
  - Log: `WARN tts.write: dropping stale chunk turn=<old_uuid> current=<new_uuid>` count climbing above ~1/turn average → the guard is firing constantly, upstream isn't aborting.
  - Metric: `voice.tts.stale_chunks_dropped_total` non-zero in steady state.
  - Behavior: user hears tail of previous answer after speaking.

---

### 2. Kokoro on 3090 Ti (CUDA stream isolation)

- **What can go wrong:** Co-locating Kokoro with the LLM on the 4090 causes 80–400ms audible TTS stutters when both kernels contend for the same SM scheduler mid-decode (design doc failure mode #8).
- **Where addressed:** Already enforced by topology. `voice-agent/docker-compose.yml` pins the `audio-service` container to GPU 1 (3090 Ti) via `CUDA_VISIBLE_DEVICES=1` / device reservation; `audio-service/src/config.py` exposes `KOKORO_DEVICE="cuda:0"` *within* that container's namespace. The LLM runs in `ghcr.io/1tommycheung/beellama-server:stable` on GPU 0 (4090).
- **How to verify (chaos test):** `bench/voice/test_chaos.py::test_tts_stutter_under_llm_load` — run **T05_concurrent_5** (5 simultaneous sessions) and assert max inter-sample gap in the TTS PCM stream < 40ms (one 24kHz frame at chunk boundary).
- **Failure indicator:**
  - Log: `kokoro.synth: chunk_gap_ms=<N>` where N > 50.
  - Metric: `voice.tts.interframe_gap_ms` p99 > 40ms.
  - `nvidia-smi -i 0` shows Kokoro processes (should show LLM only). `nvidia-smi -i 1` should show audio-service only.

---

### 3. Whisper subprocess recycle every 500 calls

- **What can go wrong:** faster-whisper (CTranslate2) accumulates VRAM fragmentation and small CPU-side leaks across long-running sessions; after a few thousand transcriptions the process slows or OOMs unpredictably.
- **Where addressed:** **Not yet implemented** (P6). Planned: `audio-service/src/stt.py` — replace the in-process `WhisperModel` with a `WhisperPool` wrapping a `multiprocessing.Process`; counter increments per `transcribe()`, restart at `recycle_every=500` (configurable via `WHISPER_RECYCLE_EVERY` in `config.py`).
- **How to verify (chaos test):** `bench/voice/test_chaos.py::test_whisper_recycle` — drive 600 short transcriptions through the same `VoiceSession`, assert (a) a process restart event is logged around call 500, (b) no transcription returns empty/error during the recycle window (handoff is graceful with a warm spare or a <500ms blocking restart), (c) RSS of the worker after restart is within 10% of cold-start RSS.
- **Failure indicator:**
  - Log: `stt.pool: recycling worker pid=<old> calls=500` should appear roughly every 500 calls. Absence under load = guard disabled.
  - Metric: `voice.stt.worker_rss_mb` should sawtooth, not grow monotonically.
  - Behavior: STT latency p95 drift > +30% from baseline over a 1-hour soak.

---

### 4. VAD `reset_states` on WebSocket disconnect

- **What can go wrong:** Silero VAD is a stateful LSTM; if the WebSocket drops mid-utterance and the same `vad` instance is reused for the next connection, leftover hidden state causes false `on_speech_start` immediately on the next session or missed endpoints.
- **Where addressed:** `audio-service/src/vad.py` — `SileroVAD.reset_states()` wrapping the ONNX model's `reset_states()`. Hooked from `audio-service/src/orchestrator.py` in the WebSocket `finally:` / disconnect handler, alongside `audio_buffer.clear()` and `turn_ctrl.new_turn()` (which invalidates any in-flight TTS bound to the dying session).
- **How to verify (chaos test):** `bench/voice/test_chaos.py::test_vad_reset_on_disconnect` — open WS, stream 0.8s of speech, force-close the WS mid-utterance, reconnect, send 1s of silence; assert no `on_speech_start` fires on the silence-only second session. Also covered indirectly by "Drop WebSocket packet → graceful continuation" in the chaos suite.
- **Failure indicator:**
  - Log: `vad: speech_start triggered with no audio energy` (RMS below threshold) on session open.
  - Metric: `voice.vad.spurious_starts_per_session` > 0 in the first 500ms of a new session.
  - Behavior: agent "answers" before user speaks on a freshly opened connection.

---

### 5. beellama priority scheduling + `max_num_seqs=2`

- **What can go wrong:** Under concurrent load, batch scheduling pads voice turns into the same batch as background/bulk requests; TTFT collapses from ~160ms to multi-second as queue depth grows.
- **Where addressed:**
  - LLM-side: deploy flag on `ghcr.io/1tommycheung/beellama-server:stable` — `--max-num-seqs 2` (set in `voice-agent/docker-compose.yml` command override for the `llm` service, or env `BEELLAMA_MAX_NUM_SEQS=2`).
  - Client-side: `audio-service/src/orchestrator.py` HTTP call to beellama sets `priority=10` in the JSON body for voice-loop requests. Any non-voice caller (eval harness, summarizer cron) must use `priority=1`. **Client-side priority tagging not yet implemented** — orchestrator currently sends no priority field.
- **How to verify (chaos test):** `bench/voice/test_concurrency.py::test_voice_priority_over_bulk` — start 1 voice session + 4 background `priority=1` long-context requests; assert voice TTFTAudio p95 stays under 900ms (vs. design budget 800ms, with 100ms slack). Also **T05_concurrent_5** for the `max_num_seqs=2` cap behavior.
- **Failure indicator:**
  - Log (beellama): `slot 0: preempted` should appear when a high-priority voice request lands during a bulk decode. Absence = priority not wired through.
  - Metric: `voice.llm.ttft_ms` p95 doubling when `voice.llm.queue_depth` > 2.
  - Behavior: voice agent "freezes" for >1s mid-conversation when an unrelated job runs on the same beellama instance.

---

## Monitoring & alerting

Run an exporter sidecar (pynvml + orchestrator-emitted metrics) and alert on:

| Signal | Threshold | Why |
|---|---|---|
| **GPU 0 (4090) VRAM used** | `> 85%` for 60s (via `pynvml.nvmlDeviceGetMemoryInfo`) | LLM headroom; >85% risks OOM on a long-context turn |
| **GPU 1 (3090 Ti) VRAM used** | `> 85%` for 60s | Whisper + Kokoro + VAD combined; spike means subprocess leak |
| **GPU 0 / GPU 1 utilization** | sustained `< 5%` while sessions active | service hung; kernel not dispatching |
| **GPU temperature (either)** | `> 83°C` | thermal throttle imminent → latency cliff |
| **`voice.session.queue_depth`** | `> 3` | scheduler saturating; new sessions will see TTFT collapse |
| **`voice.llm.ttft_ms` p95** | `> 1000ms` over 5min | LLM stack degraded (lost priority, lost MTP, KV thrash) |
| **`voice.tts.interframe_gap_ms` p99** | `> 40ms` | audio stutter — CUDA contention or Kokoro process stall |
| **`voice.tts.stale_chunks_dropped_total` rate** | `> 5/min` sustained | barge-in upstream not aborting; wasted compute |
| **`voice.stt.worker_rss_mb`** | monotonic growth over 30min, no sawtooth | Whisper recycle guard broken |
| **`voice.vad.spurious_starts_per_session`** | `> 0` on session open | VAD state leak across disconnects |
| **WebSocket disconnect rate** | `> 2%` of sessions | network or orchestrator crash loop |
| **beellama upstream 5xx rate** | `> 0.5%` of calls | LLM container unhealthy; chaos test "Kill beellama mid-stream" path |
| **End-to-end TTFTAudio p95** | `> 1200ms` (T03 budget) over 5min | composite regression — page on-call |

Alert routing: page on the last three (user-visible); ticket the rest.

Health probes:
- `GET /healthz` on audio-service — returns 200 iff Whisper, Kokoro, VAD all loaded and `turn_ctrl` responsive.
- `GET /healthz` on beellama — already provided by base image.
- Compose-level: `docker-compose ps` watchdog restarts either container on > 3 consecutive failed probes.

---

## Next-week TODO (post-P6)

Items beyond the 5-day implementation plan, sourced from the design doc's "Open questions / next investigations":

- [ ] **Streaming Whisper (vs full-utterance on VAD endpoint)** — subagent #1 estimates 100–150ms TTFTAudio win; subagent #4 picked full-utterance for P1 simplicity. A/B against the baseline.json from P5 once it lands.
- [ ] **LiveKit turn-detector as second-pass** — semantic endpoint detector layered on Silero; reduces false endpoints ~80% at ~50ms CPU cost. Sequence after Silero `on_speech_end`, before STT trigger.
- [ ] **MPS daemon on the 4090** — only needed if we co-locate a second model (reranker, embeddings) alongside the LLM. Skipped in current topology. Enable via `nvidia-cuda-mps-control -d` in the LLM container and pin both processes' `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`.
- [ ] **Multi-client load test** — extend `bench/voice/test_concurrency.py` beyond T05's 5 sessions to 10/20/50, find the knee. Requires distributed audio capture (one Python client per process) since per-session WS + audio decode is CPU-bound on the driver host.
- [ ] **"Always start with a filler" system prompt** — subagent #5's perceptual-latency hack ("hmm, ", "sure — "); A/B in T01–T03 to measure subjective TTFT without changing real TTFT.
- [ ] **Audio output codec decision** — 24kHz int16 raw PCM (current) vs Opus for browser clients. Measure end-to-end including encode/decode jitter before committing.
