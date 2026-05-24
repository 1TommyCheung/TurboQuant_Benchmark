# TurboQuant_Benchmark

Real-world inference benchmarks for **Qwen3.5-9B** on a single **RTX 4090 24GB** (WSL2), comparing TurboQuant KV variants, DFlash speculative decoding, PFlash speculative prefill, and three serving stacks (beellama, vLLM, Lucebox).

The goal: find the deployment config that maximizes decode throughput while leaving enough VRAM for a coexisting multimedia stack (Whisper STT, Kokoro TTS, VAD, SoulX video generation).

---

## 🏆 Production config (validated)

```bash
docker pull ghcr.io/1tommycheung/beellama-server:stable

docker run -d --gpus all -p 8083:8083 \
    -v ~/models:/models \
    -e MODEL_PATH=/models/Qwen3.5-9B-Q8_0.gguf \
    ghcr.io/1tommycheung/beellama-server:stable
```

| Metric | Result |
|---|---|
| Decode throughput | **71 tok/s sustained** @ 60K context |
| TTFT (60K input) | **8.96s** median (9.30s p95) |
| Accuracy (10× NIAH @ 60K) | **10/10** retrieval |
| Peak VRAM | **9.7 GB** → leaves ~14 GB headroom on 24GB card |

Image: [`ghcr.io/1tommycheung/beellama-server`](https://github.com/1TommyCheung/beellama.cpp/pkgs/container/beellama-server) (2.8 GB, CUDA 13.2.1 / Ubuntu 24.04 base, sm_89 / RTX 40xx). Source: [1TommyCheung/beellama.cpp](https://github.com/1TommyCheung/beellama.cpp) (fork of [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp)).

---

## 📊 Full speed leaderboard

All configs: Qwen3.5-9B, beellama, no DFlash, single-slot, RTX 4090 WSL2.

| Rank | Config | Decode tok/s | VRAM | TTFT | Notes |
|---|---|---|---|---|---|
| 🥇 | Q4_K_M + F16 KV @ 32K | **126** | 7.0 GB | 0.2s | Fastest overall (Q4 quality tradeoff) |
| 🥈 | Q8 + F16 KV @ 4K | 87 | 9.3 GB | <1s | Short context, max quality |
| 🥉 | **Q8 + `turbo4` KV @ 64K** | **71** | 9.7 GB | 9.0s | **Recommended production** ⭐ |
| 4 | Q8 + `turbo3` KV @ 64K | 47 | 9.9 GB | 6.6s | Mid-tier TQ variant |
| 5 | Q8 + `turbo3_tcq` KV @ 64K | 39 | 9.9 GB | 9.0s | Smallest absolute bits; TCQ decode tax |
| 6 | Q8 + DFlash @ 32K | 67-81 | 12.3 GB | 0.4s | DFlash hurts (9% accept rate) |
| 7 | Q8 + TQ3 + DFlash @ 128K | 33 | 12.8 GB | 29s | Combo loses on 9B at full KV |

---

## 🔬 Key findings

### 1. TurboQuant `turbo4` beats `turbo3_tcq` by 82% on Qwen3.5-9B

Tested all four TQ variants (`turbo3`, `turbo3_tcq`, `turbo4` on Q8 @ 64K, 10 NIAH cases each):
- `turbo4` (4-bit raw): **71 tok/s**, 9.7 GB, 10/10 accuracy ← **new production default**
- `turbo3` (3-bit raw): 47 tok/s
- `turbo3_tcq` (3-bit TCQ-encoded): 39 tok/s ← old default
- The TCQ trellis decode kernel costs ~40% throughput vs raw 3-bit at the same VRAM (overhead lost in noise at this scale)

### 2. Qwen3.5 is a hybrid attention+SSM architecture (only 8/32 layers cache KV)

Initially predicted F16 KV at 128K = ~16 GB; measured was **4.4 GB**. Found in the server log:

```
llama_kv_cache:        size = 4224 MiB (135168 cells, 8 layers, 4/1 seqs)
llama_memory_recurrent: size = 201 MiB  (4 cells,   32 layers, 4 seqs)
```

Only 8 of 32 layers store attention KV; the other 24 use a constant-size recurrent (SSM-style) state. Independently confirmed in vLLM logs: `TQ hybrid: full-attention layers [3, 7, 11, 15, 19, 23, 27, 31]`. Same family as Jamba / Falcon-H1.

**Implication:** TurboQuant KV compression saves less on Qwen3.5 than on pure-attention models (3 GB at 128K vs ~12 GB on a hypothetical Llama-3-9B). The architecture is already KV-frugal by design.

### 3. DFlash speculative decoding doesn't work properly on beellama for 9B

Despite trying Q4/Q8 target × Q8 draft × `--draft-max 8/16` × multiple contexts:
- Accept rate stuck at **4-9%** (vs z-lab's published 40-50%)
- Combo always slower than baseline at our output lengths (decode 30-80 tok/s vs 71-87 baseline)
- **Beellama's DFlash implementation lacks the block-diffusion semantics** z-lab's draft model was trained for

DFlash properly requires vLLM 0.20.1+ or SGLang with `num_speculative_tokens: 15-16`. Neither works on a 24GB GPU for Qwen3.5-9B (BF16 + DFlash drafter + hybrid arch exceeds the card's VRAM budget — we measured the engine hang at 24 GB during KV cache profiling). z-lab tested on A100 80GB / H100.

### 4. vLLM cannot combine TurboQuant + speculative decoding today

Verified against 3 implementations:

| Stack | TQ alone | Spec alone | TQ + Spec combo |
|---|---|---|---|
| vLLM v0.21.0 upstream | ✅ | ✅ | ❌ (`supports_spec_as_decode=False`) |
| vLLM + PR #40914 patch | ✅ | ✅ | ✅ MTP only, ❌ DFlash |
| Sandermage/genesis-vllm-patches | ✅ | ✅ | ✅ MTP only, Qwen3.6 only |
| mitkox/vllm-turboquant | ✅ | ❌ | ❌ no spec at all |
| **beellama** | ✅ | ✅ | ✅ **TQ + DFlash combo works** |
| Lucebox | ✅ | ✅ | ✅ + PFlash on top |

**beellama is currently the only OpenAI-compatible HTTP server with TQ + DFlash working** — and Lucebox has the only complete PFlash+TQ+DFlash stack but ships its own daemon binary.

### 5. F16 KV at 128K does NOT OOM on 9B (corrected from initial prediction)

Naive math said 16 GB KV cache at F16 + 128K context = OOM on 24GB card. Reality: 13.4 GB total (model 8 GB + KV 4.4 GB + buffers). The hybrid architecture (finding #2) explains it.

---

## 🗂 Repository structure

```
docs/
├── reports/
│   ├── 2026-05-24-final-weekend-summary.md      ← main report (start here)
│   ├── 2026-05-24-final-weekend-summary.html    ← interactive version w/ Chart.js graphs
│   └── 2026-05-24-qwen35-9b-serving-benchmark-report.{md,html}
├── deployment/
│   └── beellama-qwen35-9b.md                    ← how to use the published image
├── findings/
│   ├── 2026-05-23-mtp-dflash-thinking-mode.md
│   ├── 2026-05-24-benchmark-server-configs.md
│   ├── 2026-05-24-lucebox-rtx4090-wsl2-results.md
│   └── 2026-05-24-pflash-128k-direct-vs-http.md
├── migration/
│   └── 2026-05-25-beellama-fork-split.md        ← architecture migration plan
└── watchlist.md                                  ← upstream PRs/issues to monitor

bench_embeddings_turbo/                           ← earlier Qwen3-Embedding-8B benchmarks
```

---

## 🧰 What was tested

### Models
- **Qwen3.5-9B** — Q8_0, Q4_K_M GGUF (target); DFlash q8_0, DFlash Q4_K_M (draft); BF16 safetensors (vLLM)
- **Qwen3.6-27B Q4_K_M** — for Lucebox PFlash 27B reference
- **lovedheart/Qwen3.5-9B-FP8** — for vLLM TurboQuant testing

### Inference engines
- **beellama** (llama.cpp fork by Anbeeld) — primary
- **vLLM v0.21.0** — TurboQuant KV (PR #38479), DFlash spec decode
- **Lucebox** — PFlash + DFlash + DDTree (separate C++/CUDA daemon)

### Methodology
- 10 NIAH cases at 60K input (production validation)
- 4-config matrix at 4K context (method ablation)
- 3-config matrix at 128K context (long-ctx viability)
- 4 TurboQuant KV variants compared on same cases
- 16K-output sustained decode tests
- All requests streamed via Python httpx; per-token timestamps captured
- VRAM monitored via `nvidia-smi --query-gpu=memory.used -l 2`

---

## 🚀 Deployment

See [docs/deployment/beellama-qwen35-9b.md](docs/deployment/beellama-qwen35-9b.md) for the full integration guide. TL;DR:

```bash
# 1. Pull the pre-built image
docker pull ghcr.io/1tommycheung/beellama-server:stable

# 2. Run it (env vars override defaults; the defaults are the production config)
docker run -d --name beellama --restart unless-stopped --gpus all -p 8083:8083 \
    -v ~/models:/models \
    -e MODEL_PATH=/models/Qwen3.5-9B-Q8_0.gguf \
    ghcr.io/1tommycheung/beellama-server:stable

# 3. Use it (OpenAI Chat Completions API)
curl -N -X POST http://localhost:8083/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model":"qwen3.5-9b",
        "messages":[{"role":"user","content":"hi"}],
        "max_tokens":50, "stream":true,
        "chat_template_kwargs":{"enable_thinking":false}
    }'
```

**Image details:** 2.8 GB compressed, CUDA 13.2.1 runtime, Ubuntu 24.04 base, sm_89 (RTX 40xx). For other GPU archs, rebuild from source (instructions in the [beellama fork's docker/README.md](https://github.com/1TommyCheung/beellama.cpp/blob/stable/docker/README.md)).

---

## 🔄 What to watch upstream

Tracking 11 PRs/issues whose resolution would unlock new configurations. See [docs/watchlist.md](docs/watchlist.md). Highlights:

- **[vLLM PR #40914](https://github.com/vllm-project/vllm/pull/40914)** — TurboQuant + MTP fix (validated, awaiting merge)
- **[vLLM Issue #41559](https://github.com/vllm-project/vllm/issues/41559)** — DFlash + TQ structural fix
- **Lucebox PFlash recipe for 9B** — currently only 27B is tuned
- **Smaller DFlash draft for 9B** — would cut combo's 1.5 GB overhead

Recommended cadence: monthly check via `/schedule` cron.

---

## 🛠 Hardware tested on

- **GPU:** NVIDIA RTX 4090 24GB (sm_89, compute capability 8.9)
- **OS:** Windows 11 + WSL2 (Ubuntu 22.04)
- **Driver:** NVIDIA 596.21 (CUDA 13.2)
- **Kernel:** 6.6.87.2-microsoft-standard-WSL2

Multi-workload coexistence target: Qwen3.5-9B (~10 GB) + Whisper large-v3 (~3 GB) + Kokoro TTS (~1.5 GB) + VAD (~0.5 GB) + SoulX video gen = ~17 GB total, fits in 24 GB with margin.

---

## 📜 License

MIT. The packaged beellama binary and its llama.cpp/TurboQuant/DFlash dependencies are all MIT-licensed. NVIDIA CUDA runtime libraries shipped in the Docker image are redistributable under the NVIDIA CUDA EULA. Full SBOM at [the beellama fork's docker/SBOM.md](https://github.com/1TommyCheung/beellama.cpp/blob/stable/docker/SBOM.md).

---

## 🙏 Credits

- **[Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp)** — beellama upstream
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — TurboQuant origin
- **[z-lab/dflash](https://github.com/z-lab/dflash)** — DFlash speculative decoding research
- **[Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub)** — PFlash + DFlash + DDTree implementation
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — the foundation

This benchmark project: weekend exploration, May 2026, Tommy Cheung + Claude (Opus 4.7). Methodology, results, and recommendations are all reproducible from the published image + scripts in this repo.
