# Deploying Qwen3.5-9B on beellama — integration guide for other projects

Use this guide to plug the validated `beellama + Qwen3.5-9B-Q8 + TurboQuant3 KV @ 64K` setup
into another project as an OpenAI-compatible inference backend.

**Validated 2026-05-24 on RTX 4090 24GB / WSL2.** Numbers from `docs/reports/2026-05-24-final-weekend-summary.md`:
- 10/10 NIAH retrievals at 60K input
- TTFT median 7.58s (p95 9.44s) — **with `turbo4` KV (the production default)**
- **Decode 71 tok/s sustained at 60K** (was 39 before turbo4 swap; 87 tok/s at small context)
- Peak VRAM **9.7 GB** → leaves ~14 GB for other GPU workloads on a 24GB card

**Update 2026-05-24:** swapped default KV from `turbo3_tcq` → `turbo4` after variant sweep showed 82% decode speedup with same accuracy and same VRAM. Docker image now lives at [`ghcr.io/1tommycheung/beellama-server`](https://github.com/1TommyCheung/beellama.cpp/pkgs/container/beellama-server) (separate beellama fork, see `docs/migration/2026-05-25-beellama-fork-split.md` for the split rationale).

---

## TL;DR — minimum to integrate

Three commands, total time ~3 minutes assuming you have the model GGUF:

```bash
# 1. Pull image
docker pull ghcr.io/1tommycheung/beellama-server:stable

# 2. Run it
docker run -d --name beellama --gpus all -p 8083:8083 \
    -v ~/models:/models \
    -e MODEL_PATH=/models/Qwen3.5-9B-Q8_0.gguf \
    ghcr.io/1tommycheung/beellama-server:stable

# 3. Use it (OpenAI-compatible at http://127.0.0.1:8083/v1)
```

Always pass `chat_template_kwargs: {"enable_thinking": false}` per request to disable Qwen3.5 thinking mode.

---

## 1. Prerequisites

### Hardware

- NVIDIA GPU with **≥10 GB VRAM** for the recommended config (Q8_0 weights + turbo4 KV @ 64K)
- Tested on RTX 4090 24GB; works on RTX 4080/4070 Ti
- The published image's CUDA kernels are compiled for **sm_89 only (RTX 40xx Ada)**. For other GPUs, rebuild from source — see `docs/migration/2026-05-25-beellama-fork-split.md` Session E or the fork's `docker/README.md`
- NVIDIA driver supporting CUDA 13.2 (driver ≥ R555)

### Software / artifacts

| Item | Where | Size |
|---|---|---|
| Docker + NVIDIA Container Toolkit | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/ | n/a |
| beellama-server image (pre-built) | `ghcr.io/1tommycheung/beellama-server:stable` | 2.8 GB |
| Qwen3.5-9B Q8_0 GGUF | local file or HuggingFace auto-download | 9.5 GB |
| (optional) DFlash draft | `Qwen3.5-9B-DFlash-q8_0.gguf` | 1.1 GB |

**Critical:** put model files on **native ext4** (e.g., `~/models/`). NTFS via WSL2 `/mnt/` triples model load time. See `docs/findings/2026-05-24-pflash-128k-direct-vs-http.md`.

### Fetch the model

```bash
mkdir -p ~/models
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q8_0.gguf --local-dir ~/models/

# (optional) DFlash draft for speculative decoding — NOT recommended on 24GB GPUs;
# accept rate stuck at ~9% in our testing. See report Section §Group A/B/C.
# hf download <repo>/Qwen3.5-9B-DFlash-q8_0.gguf --local-dir ~/models/
```

Or skip local file mount entirely — let the container auto-download from HuggingFace on first start:
```bash
docker run --gpus all -p 8083:8083 -v hf-cache:/root/.cache/huggingface \
    -e HF_MODEL_REPO=unsloth/Qwen3.5-9B-GGUF \
    -e HF_MODEL_FILE=Qwen3.5-9B-Q8_0.gguf \
    ghcr.io/1tommycheung/beellama-server:stable
```

---

## 2. Start the server

### Recommended production config (validated) — via Docker

```bash
docker run -d --name beellama --restart unless-stopped \
    --gpus all -p 8083:8083 \
    -v ~/models:/models \
    -e MODEL_PATH=/models/Qwen3.5-9B-Q8_0.gguf \
    -e CONTEXT_SIZE=65536 \
    -e CACHE_TYPE_K=turbo4 -e CACHE_TYPE_V=turbo4 \
    ghcr.io/1tommycheung/beellama-server:stable
```

All other defaults (`-fa on`, `-ngl 99`, `--reasoning-format none`, port 8083) are baked into the image.

### Equivalent bare-metal command (if you build from source)

```bash
~/beellama.cpp/build/bin/llama-server \
    -m ~/models/Qwen3.5-9B-Q8_0.gguf \
    --host 127.0.0.1 --port 8083 \
    -c 65536 -ngl 99 -fa on \
    -ctk turbo3_tcq -ctv turbo3_tcq \
    --reasoning-format none
```

| Flag | Meaning |
|---|---|
| `-m <path>` | Model GGUF on ext4 |
| `--host --port` | Listen address (`127.0.0.1` for local-only; `0.0.0.0` for LAN) |
| `-c 65536` | 64K token context window |
| `-ngl 99` | Offload all layers to GPU |
| `-fa on` | Flash-attention (mandatory for the KV-compactness we measured) |
| `-ctk turbo3_tcq -ctv turbo3_tcq` | TurboQuant3 KV cache (~4× compression on attention layers) |
| `--reasoning-format none` | Disable thinking mode parsing server-side (also pass per-request, see §3) |

**Don't add `-md <draft>` and `--spec-branch-budget` for 9B real-time use** — DFlash adds ~1.5 GB VRAM and slows decode at short outputs (validated in Group A/B of the benchmark). Only enable speculative decoding if you sustain >1K-token outputs at long context.

### Wait for ready (this is the gotcha)

`llama-server`'s `/health` endpoint returns 200 **before** the model is loaded. Don't trust it. Probe the actual completion endpoint:

```bash
until curl -fs -o /dev/null -X POST "http://127.0.0.1:8083/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'; do
    sleep 2
done
echo "server is actually ready"
```

Expect ~30-60s to be ready (first run cold; subsequent runs faster).

---

## 3. Client integration

The server is OpenAI Chat Completions API-compatible. **Always pass `chat_template_kwargs.enable_thinking=false`** to prevent the model from spending tokens in `<think>...</think>` chains (which also kills speculative decoding if you enable it).

### Python — OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8083/v1",
    api_key="not-needed-but-required-by-sdk",
)

response = client.chat.completions.create(
    model="qwen3.5-9b",  # any string; beellama ignores model name with single-model serve
    messages=[{"role": "user", "content": "Summarize the discovery of penicillin."}],
    max_tokens=512,
    temperature=0.0,
    stream=True,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### Python — raw httpx (lower overhead, used in our benchmarks)

```python
import httpx, json

body = {
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256,
    "temperature": 0.0,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": False},
}
with httpx.stream("POST", "http://127.0.0.1:8083/v1/chat/completions",
                  json=body, timeout=600.0) as r:
    for line in r.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        delta = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if delta:
            print(delta, end="", flush=True)
```

### curl one-liner (smoke test)

```bash
curl -N -X POST "http://127.0.0.1:8083/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model":"qwen3.5-9b",
        "messages":[{"role":"user","content":"In one sentence: what is RAG?"}],
        "max_tokens":80,
        "temperature":0,
        "stream":true,
        "chat_template_kwargs":{"enable_thinking":false}
    }'
```

### Other SDKs (LangChain, LlamaIndex, etc.)

Any SDK that takes an OpenAI-compatible `base_url` works. Point it at `http://127.0.0.1:8083/v1`. For SDKs that don't expose `chat_template_kwargs` directly, either:
- Use the raw HTTP client and pass `chat_template_kwargs` in the request body, OR
- Keep the server flag `--reasoning-format none` and add `--chat-template-kwargs '{"enable_thinking":false}'` to the **server** startup (some beellama builds support this).

---

## 4. Operational setup

### systemd unit (Linux, run as user)

`~/.config/systemd/user/qwen35-9b.service`:

```ini
[Unit]
Description=beellama Qwen3.5-9B inference server
After=network.target

[Service]
Type=simple
ExecStart=/home/%u/beellama.cpp/build/bin/llama-server \
    -m /home/%u/models/Qwen3.5-9B-Q8_0.gguf \
    --host 127.0.0.1 --port 8083 \
    -c 65536 -ngl 99 -fa on \
    -ctk turbo3_tcq -ctv turbo3_tcq \
    --reasoning-format none
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/%u/logs/qwen35-9b.log
StandardError=append:/home/%u/logs/qwen35-9b.log

[Install]
WantedBy=default.target
```

Enable and start:
```bash
mkdir -p ~/logs
systemctl --user daemon-reload
systemctl --user enable --now qwen35-9b
systemctl --user status qwen35-9b
journalctl --user -u qwen35-9b -f
```

For WSL2, you may need `loginctl enable-linger $USER` to keep the user service alive after logout.

### Docker (if you prefer containers)

beellama doesn't ship a public Docker image. If you containerize:
- Base on `nvidia/cuda:12.6.0-runtime-ubuntu22.04`
- Build llama-server inside or copy the binary in
- Mount your model dir as read-only volume
- `docker run --gpus all -v ~/models:/models:ro -p 8083:8083 <image>` with the same flags

### GPU sharing with other workloads

The recommended config uses ~10 GB. To leave room for Whisper/TTS/video gen:

| Workload | Typical VRAM | Tip |
|---|---|---|
| Whisper large-v3 | 3 GB | Use `faster-whisper` for half the VRAM |
| Kokoro TTS | 1-2 GB | |
| VAD (Silero/webrtcvad) | 0.5 GB or CPU | Run on CPU if possible |
| **9B inference (this server)** | **10 GB** | Fixed |
| **Headroom (24 GB - 13 GB used)** | **~11 GB** | For video gen or larger speech models |

Don't run inference processes that *also* call `cudaMalloc(0.9 * total)`. vLLM and SGLang are particularly greedy; they'll grab whatever you let them. beellama only takes what it needs.

### Logging

beellama logs to stderr. Per-request entries look like:
```
slot launch_slot_: id 0 | task 3 | processing task
slot print_timing: id 0 | task 3 |
prompt eval time = 5234.11 ms / 5012 tokens (1.04 ms per token, 957.7 tokens per second)
       eval time = 1456.32 ms / 128 tokens (11.38 ms per token, 87.9 tokens per second)
      total time = 6690.43 ms / 5140 tokens
```

Useful for production monitoring: parse these timing lines to feed Prometheus/Grafana.

### Health & readiness probes

```bash
# Liveness (fast, false-positive risk before model loads)
curl -fs http://127.0.0.1:8083/health

# Readiness (slow, true-positive)
curl -fs -X POST http://127.0.0.1:8083/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
```

Use readiness probe for load balancer / Kubernetes; liveness probe is fine for "is the process still alive."

---

## 5. Tuning knobs (when to deviate from the recommended config)

### When you need MORE context (up to 128K)

Change `-c 65536` to `-c 131072`. Validated to work; expect:
- Peak VRAM rises ~1.5 GB (to ~11.5 GB)
- TTFT scales linearly with prompt length (60K = 9s → 128K = ~20s)
- Decode tok/s drops slightly (full KV reads are heavier)

### When you need LESS VRAM (sub-8 GB)

Drop to Q4_K_M weights:
```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models/
# Then: -m ~/models/Qwen3.5-9B-Q4_K_M.gguf
```
- Weights ~5.5 GB (vs 9.5 GB Q8)
- Total peak ~7 GB at 64K context with TQ3 KV
- Quality cost: ~1-2% perplexity, usually imperceptible for chat/agentic tasks

### When you need FASTER decode (give up VRAM)

Drop TurboQuant, use F16 KV: `-ctk f16 -ctv f16`
- Decode rises from ~39 tok/s to ~75 tok/s at 60K
- VRAM rises ~3 GB (F16 KV is ~4× bigger than TQ3 on the 8 attention layers; the other 24 layers are recurrent and unaffected)

### When you have long generation (>1K tokens output)

Enable DFlash speculative decoding:
```bash
# Add to server CLI:
    -md ~/models/Qwen3.5-9B-DFlash-q8_0.gguf \
    -ctkd turbo3_tcq -ctvd turbo3_tcq \
    --spec-branch-budget 16 --draft-max 8 --draft-min 1
```
- Costs ~1.5 GB VRAM
- Net win depends on acceptance rate (we measured 9% @ 4K → 45% @ 128K)
- Generally **not worth it on 9B at <500 token outputs** per our Group A/B/C data

### When you have multiple concurrent users

beellama is **single-slot** (`-np 1` effective). For multi-user, switch to vLLM:
- `vllm serve <model> --max-num-seqs 16` etc.
- vLLM has higher base VRAM (~17 GB) but does continuous batching
- See vLLM section in `docs/reports/2026-05-24-final-weekend-summary.md` for the configured invocation

---

## 6. Multimedia coexistence pattern

If running alongside Whisper + Kokoro + VAD on the same GPU:

```
┌─────────────────────────────────────────────────┐
│ Process A: Whisper streaming STT  (3 GB)        │
│ Process B: Kokoro TTS              (1.5 GB)     │
│ Process C: VAD (or CPU)            (0.5 GB)     │
│ Process D: beellama qwen3.5-9b     (10 GB)      │
│ ─────────────────────────────────────────────── │
│ Total: ~15 GB ── 9 GB headroom on 24 GB         │
└─────────────────────────────────────────────────┘
```

**Tips:**
- Start beellama LAST so it picks up the remaining VRAM
- If you need to add a video gen workload (e.g., SoulX), drop beellama to Q4_K_M target to free 4 GB
- Avoid running multiple inference processes with auto-grow VRAM behavior (vLLM, SGLang) — they'll squeeze each other out
- Profile real usage with `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` while everything is running

---

## 7. Troubleshooting (errors we actually hit)

| Symptom | Cause | Fix |
|---|---|---|
| `error loading model vocabulary: cannot find tokenizer merges in model file` | GGUF was converted without tokenizer.merges | Re-download a known-good GGUF; for DFlash drafts use Q8_0 (Q4_K_M variant we tried was broken) |
| Server returns HTTP 400 with `exceeds the available context size` | Prompt (post chat-template) > `-c N` | Bump `-c` by 1-2K, or trim the prompt. Chat template adds ~10-20 tokens overhead |
| Server returns instantly with no content | Likely false-positive on `/health` — model not actually loaded | Use the readiness probe from §2 |
| Model thinks aloud in `<think>...</think>` blocks | Thinking mode is on | Pass `chat_template_kwargs:{"enable_thinking":false}` per request AND `--reasoning-format none` on the server |
| 9% draft acceptance rate, decode slower with spec | DFlash draft mismatched or thinking mode bleeding in | Verify draft GGUF is Q8_0 variant, disable thinking, ensure `--reasoning-format none` |
| Model load takes 5+ minutes | Loading from NTFS (`/mnt/...`) on WSL2 | Move GGUF to ext4 (`~/models/`). 3-5× speedup |
| `pkill -f llama-server` killed your terminal | pgrep matched your shell's PATH/env containing "llama-server" | Kill by specific PID: `pgrep -x llama-server \| xargs kill` |

---

## 8. Performance reference (RTX 4090 24GB, WSL2)

From `docs/reports/2026-05-24-final-weekend-summary.md`. Use these as sanity checks for your deployment.

| Context | Config | TTFT | Decode tok/s | Peak VRAM |
|---|---|---|---|---|
| 4K | F16 KV (no TQ, no spec) | <1s | 87 | 9.3 GB |
| 4K | TQ3 KV | <1s | 84 | 9.2 GB |
| **60K** | **turbo4 KV ⭐ (recommended)** | **7.6s** | **71** | **9.7 GB** |
| 60K | turbo3 KV | 6.6s | 47 | 9.9 GB |
| 60K | turbo3_tcq KV (old default) | 9.0s | 39 | 9.9 GB |
| 128K | F16 KV | 21s | 60 | 13.4 GB |
| 128K | turbo3_tcq KV (no spec) | ~21s | ~70 | ~10.5 GB |
| 128K | turbo3_tcq + DFlash combo | 29s | 33 | 12.8 GB |

Real-time interactivity (voice agent loop with short prompts <4K):
- TTFT < 200 ms
- Decode ~85 tok/s = 12 ms/token
- Total to first audio chunk: ~1 sec including Whisper + Kokoro

---

## 9. Architecture note (don't be surprised)

**Qwen3.5 is a hybrid attention + recurrent (SSM-style) architecture.** Only 8 of its 32 layers store a token-indexed KV cache; the other 24 use a constant-size recurrent state regardless of context length. This means:

- KV cache stays small even at 128K (~4 GB F16, ~1 GB TQ3)
- TurboQuant savings are smaller in absolute terms than on pure-attention models like Llama-3
- Recurrent state is ~200 MB and doesn't compress with TQ

Verify on your install by checking the startup log:
```
llama_kv_cache: size = X MiB (N cells, K layers, ...)
```
If `K < n_layer`, it's hybrid (Qwen3.5 → `K=8`, `n_layer=32`).

---

## 10. References

- Full benchmark methodology + raw numbers: `docs/reports/2026-05-24-final-weekend-summary.md`
- HTML version with charts: `docs/reports/2026-05-24-final-weekend-summary.html`
- vLLM TurboQuant constraints: `docs/findings/...` and the project memory `reference_vllm_turboquant`
- PFlash vs HTTP findings: `docs/findings/2026-05-24-pflash-128k-direct-vs-http.md`
- Upstream watchlist (PRs/issues to monitor): `docs/watchlist.md`
