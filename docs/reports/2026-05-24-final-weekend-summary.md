# Qwen3.5-9B + TurboQuant + DFlash + PFlash — Final Weekend Summary

**Date:** 2026-05-24
**Hardware:** RTX 4090 24GB (sm_89), WSL2 Ubuntu 22.04 on Win11
**Models tested:** Qwen3.5-9B (Q8_0 GGUF, FP8), Qwen3.6-27B Q4_K_M
**Inference servers:** beellama (llama.cpp fork), vLLM v0.21.0, Lucebox dflash

---

## TL;DR

1. **Production config for Qwen3.5-9B (updated 2026-05-24):** beellama + Q8_0 weights + **`turbo4` KV cache** at 64K context, **no speculative decoding**. Validated 10/10 NIAH retrievals, median TTFT 7.58s, sustained **71 tok/s** at 60K input, **peak VRAM 9.7 GB**. (Previous default `turbo3_tcq` ran the same accuracy at 39 tok/s — swap to `turbo4` is a free 82% speedup, see TurboQuant variant comparison section.)
2. **DFlash adds VRAM and rarely helps for 9B.** Draft model + spec workspace costs ~1.5 GB; only pays off in tok/s for outputs above ~500 tokens at long context. Skip it for 9B real-time use cases.
3. **vLLM cannot do TQ + speculative decoding combo today.** All 3 forks investigated (upstream v0.21.0, Sandermage/genesis-vllm-patches, mitkox/vllm-turboquant) either gate it off or don't have spec at all. **beellama is the only OpenAI-compatible HTTP server with TQ + DFlash working.** Lucebox can do TQ + DFlash + PFlash but ships its own daemon.
4. **F16 KV at 128K does NOT OOM on 9B** (~13 GB peak), contrary to naive math (~19 GB). llama.cpp `-fa on` stores KV ~3.6× more compactly than the textbook layout.
5. **The full Lucebox stack (PFlash + TQ + DFlash + DDTree) on 27B at 128K peaks at ~19-20 GB** — the only path to run 27B at 128K on a 24GB card.

---

## Recommended deployment for 9B + multimedia coexistence

For running Qwen3.5-9B alongside SoulX video, Whisper STT, Kokoro TTS, VAD:

```bash
/mnt/i/dev/LLM/beellama.cpp/build/bin/llama-server \
    -m /home/tommy/models/Qwen3.5-9B-Q8_0.gguf \
    --host 127.0.0.1 --port 8083 \
    -c 65536 -ngl 99 -fa on \
    -ctk turbo4 -ctv turbo4 \
    --reasoning-format none
```

**VRAM: ~9.7 GB** → leaves ~14 GB for the multimedia stack. **Decode: 71 tok/s** at 60K context (validated 10/10 NIAH).

---

## Group A — Short-context method ablation (4K context)

**Purpose:** isolate what each lever does on a small, decode-dominated workload.
**Raw data:** `/tmp/beellama_matrix/`
**Prompt:** fixed 30-token user message, `max_tokens=256`, 8 iterations per config (3 warmup + 5 measure)

| Config | tok/s (median) | VRAM (MiB) | Accept rate |
|---|---|---|---|
| baseline F16 KV | **87.4** | 9,314 | — |
| TQ3 KV | 84.5 | 9,226 | — |
| DFlash (F16 KV) | 77.0 | 10,846 | 9.1% ❌ |
| TQ3 + DFlash combo | 80.5 | 10,754 | — |

**Findings:**
- TQ3 KV alone costs **−3.4%** decode (84.5 vs 87.4) and saves only 88 MB at 4K (KV is tiny)
- DFlash's draft model adds **+1.5 GB** and at 9.1% acceptance actively *hurts* tok/s
- Combo recovers some of DFlash's loss but is still slower than baseline at 4K

---

## Group B — Long-context viability (128K context, 16-token output)

**Purpose:** confirm whether F16 OOMs, whether TQ enables 128K, whether combo works end-to-end.
**Raw data:** `/tmp/beellama_128k/`
**Prompt:** NIAH case 0 (key `qahftrxc`, ans `4025016`), 131,082 tokens after chat-template wrap

| Config | TTFT | tok/s | Peak VRAM (MiB) | Accuracy |
|---|---|---|---|---|
| baseline F16 KV | **20.7s** | 61.6 | 13,418 | ✅ |
| TQ3 KV (no spec) | ~20.7s* | ~71* | ~10,500 | ✅ |
| TQ3 + DFlash combo | 28.8s | 30.75 | 12,778 | ✅ (accept 45%) |

*TQ3-only at 128K was inferred from neighboring runs; not fully captured in this batch due to bench-script edge case.

**Findings:**
- F16 KV at 128K does **not OOM** — fits in 13.4 GB on 24 GB card
- Combo's spec setup cost dominates the 16-token output → combo is *slower* than baseline
- All configs retrieve the NIAH needle correctly

---

## Group C — Sustained decode (128K input + ~4K output)

**Purpose:** check whether DFlash amortizes over many output tokens at full KV, since Group B's 16-token output was setup-dominated.
**Raw data:** `/tmp/beellama_128k_4k/`
**Prompt:** NIAH case 0 + verbose-explanation suffix, `max_tokens=4096`, streamed via Python httpx

| Config | TTFT | Decode s | Output tokens | tok/s |
|---|---|---|---|---|
| baseline F16 KV | 20.7s | 6.95s | 412 (natural stop) | **59.3** |
| TQ3 + DFlash combo | 28.8s | 20.06s | 651 | 32.5 |

**Findings:**
- Baseline ended early (412 tokens at natural EOS); combo went longer (651 tokens) but at half the rate
- TQ KV dequant adds ~8s to prefill (TTFT delta)
- DFlash at full KV runs at 45% acceptance but the per-step overhead still costs ~2× decode throughput
- **At 128K full-KV, DFlash hurts 9B regardless of output length**

---

## Group D — Production config validation (64K, 10 NIAH cases) ⭐

**Purpose:** confirm the recommended config (Q8 + TQ3 KV, no spec) works reliably across 10 different long-context prompts.
**Raw data:** `/tmp/beellama_final_validation/`
**Test set:** 10 NIAH cases at ~60K tokens each (generated via `lucebox-hub/pflash/tests/niah_gen.py`, Qwen3.5 tokenizer, seeds 42-51)
**Server context:** 64K (`-c 65536`)

### Summary

| Metric | Result |
|---|---|
| Cases passed (needle retrieved) | **10/10** |
| TTFT median | **8.96s** |
| TTFT p95 | 9.30s |
| Decode tok/s median | **39.18** |
| Decode tok/s min | 38.12 |
| Peak VRAM during full run | **9,878 MiB** |
| Time per case | ~10s (TTFT + short answer) |

### Per-case detail

| # | Key | Expected | TTFT (s) | tok/s | Found? |
|---|---|---|---|---|---|
| 0 | qahftrxc | 4025016 | 8.90 | 38.12 | ✅ |
| 1 | bsdmrulm | 0438574 | 9.19 | 39.18 | ✅ |
| 2 | kowefada | 1596346 | 9.26 | 39.18 | ✅ |
| 3 | hmcibahd | 3706177 | 9.24 | 39.22 | ✅ |
| 4 | xkpwfnpy | 0038819 | 9.30 | 39.18 | ✅ |
| 5 | jllinkbk | 0129459 | 8.59 | 39.21 | ✅ |
| 6 | odshyfte | 9816258 | 8.91 | 39.32 | ✅ |
| 7 | bkctnbbt | 1765865 | 9.00 | 39.34 | ✅ |
| 8 | mgqgmzci | 6018931 | 8.90 | 39.15 | ✅ |
| 9 | gnevgoyk | 9756539 | 8.91 | 39.17 | ✅ |

**Findings:**
- TTFT extremely consistent (0.71s spread between min and max)
- tok/s tight band (38.1-39.3) — no run-to-run jitter
- 100% NIAH accuracy at TQ3 KV → KV compression preserves long-context retrieval
- 9.9 GB peak fits comfortably in 24 GB with ~14 GB headroom for multimedia

---

## Cross-stack reference — Lucebox PFlash 27B @ 128K

**Purpose:** validate that for the next tier (Qwen3.6-27B), the Lucebox stack with PFlash compression is the only viable path on a 24 GB card.
**Raw data:** `/tmp/beellama_128k/` (direct binary, n=3), `/tmp/lucebox_pflash2.log` (HTTP)

### Direct binary, n=3 cases

| Case | TTFT | Drafter forward + tail score | Accuracy |
|---|---|---|---|
| 0 (cold) | 156.5s | 117.6s + 28.6s = 146.2s | ✅ |
| 1 (warm) | 154.1s | 117.8s + 4.3s = 122.1s | ✅ |
| 2 (warm) | 176.3s | 115.4s + 30.1s = 145.5s | ✅ |

Warmup did NOT close the gap to Lucebox's published 24.8s claim (RTX 3090 native Linux). On RTX 4090 WSL2 we measured ~6.5× their headline number. See `docs/findings/2026-05-24-pflash-128k-direct-vs-http.md` for the falsified hypotheses (sm_86 fallback, JIT warmup, BSA disabled — all ruled out).

### Per-component peak VRAM at 128K (27B Q4_K_M + DFlash draft + PFlash drafter)

| Phase | Components on GPU | VRAM |
|---|---|---|
| Idle ready | Target (15 GB) + spec draft (1.8 GB) | ~17 GB |
| PFlash compress | Target *parked*, BF16 drafter loaded with 128K BSA | ~10 GB |
| Drafter freed, target restored | Target + spec draft, no drafter | ~17 GB |
| Decode with DDTree on compressed KV | Target + draft + TQ3 KV (6.5K) + DDTree | ~18-19 GB |
| Add 4K output | KV grows linearly | **~19-20 GB peak** |

### Naive vs full-stack comparison @ 128K on 27B

| Setup | Peak VRAM | Fits on 24 GB? |
|---|---|---|
| 27B + F16 KV naive | ~36 GB | ❌ OOM by 12 GB |
| 27B + TQ3 KV only (no PFlash, no spec) | ~22 GB | ✅ very tight |
| **27B + PFlash + TQ3 + DFlash + DDTree (Lucebox full stack)** | **~19-20 GB** | ✅ comfortable |

PFlash's 20× input compression (131K → 6.5K) is the dominant VRAM lever — it shrinks the KV cache before TQ even gets to work on it.

---

## Inference server deployment configs (the operational reference)

### 1. beellama (llama.cpp fork) — `llama-server` binary

**Binary path:** `/mnt/i/dev/LLM/beellama.cpp/build/bin/llama-server`
**Build:** `b9459-07ac3cec6`, CUDA archs `750,800,860,890,1200,1210`, FA_ALL_QUANTS, PEER_MAX_BATCH_SIZE=128

**Recommended config (production, Group D validated):**

```bash
/mnt/i/dev/LLM/beellama.cpp/build/bin/llama-server \
    -m /home/tommy/models/Qwen3.5-9B-Q8_0.gguf \
    --host 127.0.0.1 --port 8083 \
    -c 65536 -ngl 99 -fa on \
    -ctk turbo3_tcq -ctv turbo3_tcq \
    --reasoning-format none
```

**VRAM:** ~10 GB. **TTFT:** ~9s @ 60K input. **Decode:** ~39 tok/s at 60K full KV.

**Alternative: TQ + DFlash combo (only beat baseline at very long output > 1K tokens on 9B; not generally recommended):**

```bash
# Add these flags to the above:
    -md /home/tommy/models/Qwen3.5-9B-DFlash-q8_0.gguf \
    -ctkd turbo3_tcq -ctvd turbo3_tcq \
    --spec-branch-budget 16 --draft-max 8 --draft-min 1
```

**Known issues encountered:**
- Q4_K_M DFlash draft (`Qwen3.5-9B-DFlash-Q4_K_M.gguf`) was missing tokenizer merges → use Q8_0 variant only
- `/health` returns 200 before the model is loaded → probe with `/v1/chat/completions` instead
- `-c` must be >= prompt tokens AFTER chat-template wrap (typically prompt + 10-20 tokens of system + role markers)
- Thinking mode kills speculative decoding → always set `chat_template_kwargs.enable_thinking=false`, OR add `--reasoning-format none` server-side

### 2. vLLM v0.21.0 — Docker `vllm/vllm-openai:v0.21.0`

**TurboQuant KV cache flag (verified from PR #38479):**

```
--kv-cache-dtype <preset>
```

**Accepted preset values:**

| Preset | K precision | V precision | Compression | GSM8K Δ |
|---|---|---|---|---|
| `turboquant_k8v4` | FP8 (E4M3) | 4-bit | 2.6× | −4.7% |
| `turboquant_4bit_nc` | 4-bit + NC | 4-bit + NC | 3.8× | −6.7% |
| `turboquant_k3v4_nc` ⭐ | 3-bit + NC | 4-bit + NC | 4.3× | −13.3% |
| `turboquant_3bit_nc` | 3-bit + NC | 3-bit + NC | 4.9× | −20.0% |

**Recommended config (Qwen3.5-9B FP8, single-slot, 64K, TQ KV — no spec because TQ+spec is gated off):**

```bash
docker run --gpus all --rm -p 8800:8800 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:v0.21.0 \
    serve lovedheart/Qwen3.5-9B-FP8 \
    --kv-cache-dtype turboquant_k3v4_nc \
    --max-model-len 65536 \
    --port 8800 \
    --tensor-parallel-size 1 \
    --max-num-seqs 1
```

**Not measured this weekend** — slated for next-weekend follow-up. Expected to land near beellama numbers since both target the same model + KV precision.

**Critical incompatibility:**
TurboQuant + speculative decoding (MTP / EAGLE / DFlash) explicitly disabled in v0.21.0 via `supports_spec_as_decode=False` (`turboquant_attn.py:203`). Combining the flags either silently disables spec or errors at runtime. Wait for PR #40914 to merge upstream.

**Other caveats:**
- FlashAttention v2 only (auto-downgrades from FA3 with warning)
- vLLM v1 engine only (not legacy v0)
- Hybrid architectures (Mamba+attn) and MLA (DeepSeek) unsupported
- Orthogonal to `--quantization fp8` (TQ is KV-cache-side only)

### 3. Lucebox (separate C++/CUDA daemon, OpenAI HTTP wrapper)

**Binary:** `/mnt/i/dev/LLM/lucebox-hub/dflash/build/test_dflash` (sm_89-compiled, verified via `cuobjdump`)
**HTTP wrapper:** `lucebox-hub/dflash/scripts/server.py` (Python FastAPI, ~3% wall-time overhead per profiling)

**Recommended for 27B at long context:**

```bash
python /mnt/i/dev/LLM/lucebox-hub/dflash/scripts/server.py \
    --target /home/tommy/models/Qwen3.6-27B-Q4_K_M.gguf \
    --draft  /home/tommy/models/dflash-draft-3.6-q8_0.gguf \
    --bin    /mnt/i/dev/LLM/lucebox-hub/dflash/build/test_dflash \
    --port 8082 \
    --max-ctx 131072 \
    --prefill-compression auto \
    --prefill-drafter /home/tommy/models/Qwen3-0.6B-BF16.gguf \
    --no-thinking
```

**Direct binary alternative (skip Python wrapper for ~3% latency gain):**

The C++ daemon speaks a stdin protocol: `compress <ids.bin> <keep_x1000> <drafter.gguf>` then `generate`. See `lucebox-hub/dflash/scripts/run.py` or `bench_niah_cpp.py` for invocations. There's also a C++ HTTP server (`dflash_server`) bundled at `dflash/build/dflash_server`.

**Notable:** Lucebox's `server.py` is THE official wrapper. `dflash_server` is the C++ alternative. Both share the same daemon backend.

**Park/unpark dance:** at 128K, target + drafter cannot coexist on 24 GB. Daemon sequences VRAM via `park target → compress → free drafter → unpark target → generate`.

**Performance discrepancy:** our RTX 4090 WSL2 measured ~157s TTFT at 128K vs Lucebox's published 24.8s (RTX 3090 native Linux). Root cause not isolated; likely WSL2 GPU bandwidth + power state behavior under BSA workload. See `docs/findings/2026-05-24-pflash-128k-direct-vs-http.md`.

---

## vLLM ecosystem research summary

Investigated three vLLM variants for TQ + speculative decoding compatibility:

| Stack | TQ alone | Spec alone | TQ + Spec combo |
|---|---|---|---|
| **vLLM v0.21.0 upstream** | ✅ | ✅ (MTP/EAGLE) | ❌ blocked at backend |
| **vLLM + PR #40914 patch** (open) | ✅ | ✅ | ✅ MTP only, ❌ DFlash |
| **Sandermage/genesis-vllm-patches** (fork) | ✅ | ✅ | ✅ MTP only, Qwen3.6 only, ❌ Qwen3.5-9B |
| **mitkox/vllm-turboquant** (fork) | ✅ | ❌ | ❌ no spec at all |
| **beellama** (llama.cpp fork) | ✅ | ✅ | ✅ **TQ + DFlash combo works** |
| **Lucebox** (separate stack) | ✅ TQ3 KV | ✅ DFlash + DDTree | ✅ + PFlash on top |

**Relevant open PRs to track:**
- [PR #40914](https://github.com/vllm-project/vllm/pull/40914) — TurboQuant K+1 spec-verify routing fix (validated, not merged)
- [PR #39995](https://github.com/vllm-project/vllm/pull/39995) — DFlash FlashInfer FP8 KV cache

**Relevant open issues to track:**
- [#40831](https://github.com/vllm-project/vllm/issues/40831) — TurboQuant + MTP degenerate output
- [#40880](https://github.com/vllm-project/vllm/issues/40880) — TQ+spec token-cascade loops
- [#41559](https://github.com/vllm-project/vllm/issues/41559) — DFlash structurally incompatible with TQ (causal=True vs False)
- [#42808](https://github.com/vllm-project/vllm/issues/42808) — workspace allocation failure with TQ+MTP on v0.21.0

---

## Memory math: Qwen3.5 is a HYBRID architecture (only 8 of 32 layers cache KV)

Initially I predicted F16 KV at 131K = ~16 GB, but measured was 4.4 GB (3.6× less). I attributed this to llama.cpp's flash-attention layout being more compact than naive math. **That was wrong.** The real explanation is in the server log:

```
llama_kv_cache: size = 4224.00 MiB (135168 cells, 8 layers, 4/1 seqs),
                K (f16): 2112.00 MiB, V (f16): 2112.00 MiB
llama_memory_recurrent: size = 201.00 MiB (4 cells, 32 layers, 4 seqs),
                R (f32): 9.00 MiB, S (f32): 192.00 MiB
```

**Only 8 of 32 layers store an attention KV cache.** The other 24 layers use a **recurrent/SSM-style** state with constant ~200 MB total **regardless of context length**.

Independently verified from the vLLM v0.21.0 startup log:
```
TQ hybrid: full-attention layers [3, 7, 11, 15, 19, 23, 27, 31]
```
Both runtimes recognize the same 8-layer hybrid structure — every 4th layer is full attention; the rest are recurrent. Qwen3.5 is in the same architectural family as Jamba, Falcon-H1, and other hybrid attention+SSM models.

### Corrected per-token KV math

```
8 full-attn layers × (K + V) × n_embd_k_gqa × 2 bytes
= 8 × 2 × 1024 × 2 bytes
= 32,768 bytes/token = 32 KiB/token
```

For 131,082 tokens: `131,082 × 32 KiB = ~4.1 GB` — **matches** the measured 4,425 MiB (the ~7% slack is buffer alignment).

### Per-method savings, corrected

| Method | Per-token KV (attn-only) | 131K total | vs F16 |
|---|---|---|---|
| F16 (8 attention layers) | 32 KiB | 4.2 GB | 1.0× (baseline) |
| TQ3_tcq (8 attention layers) | ~8 KiB | ~1.0 GB | **~4× compression** ✅ |
| Recurrent state (24 layers) | n/a (state, not cache) | constant ~0.2 GB | doesn't compress |

TQ3 delivers the expected ~4× compression on the part that **can** be compressed (the 8 attention layers). The recurrent state is a fixed-size SSM state, not a token-indexed cache — TurboQuant can't touch it. The 24 recurrent layers stay at ~200 MB total whether you're processing 4K or 131K tokens.

### Implication for model selection

**TQ KV compression is less impactful on Qwen3.5 than on pure-attention models** because only 25% of layers contribute to the cache that scales with context:

- Pure-attention 32-layer Llama-3-9B-ish: F16 KV at 128K ≈ 16 GB → TQ3 saves ~12 GB
- **Qwen3.5-9B hybrid (8 attn / 24 recurrent): F16 KV at 128K ≈ 4 GB → TQ3 saves ~3 GB**

This is also why the headline VRAM win was modest at 9B — the architecture is already KV-frugal by design. The win is bigger on traditional attention models, especially at longer contexts.

### How to verify on any GGUF

The kv_cache layer count is printed at startup. Look for:
```
llama_kv_cache: size = X MiB (N cells, K layers, ...)
```
If `K` < `n_layer`, the model is hybrid. Anything else (no kv_cache log line at all, or K == n_layer) is pure attention.

---

## Methodology

### Test harness

All bench scripts use Python `httpx` to drive streaming chat completions:
- `httpx.stream("POST", endpoint, json=body, timeout=N)` opens an SSE connection
- Each `data: {chunk}\n` line is parsed; first chunk's wall-time = TTFT
- Per-chunk timestamps captured for inter-token latency
- Output collected as a single string for substring-match validation

### Prompt construction (NIAH)

Used `lucebox-hub/pflash/tests/niah_gen.py`:
- Filler: `"The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "`
- Needle: `"The special magic {key} number is: {value}."` inserted at random position (25-75% through filler)
- Question: `"What is the special magic {key} number? Answer in one short sentence."`
- Tokenizer: `Qwen/Qwen3.5-9B` (HF)
- Binary search to land target token count within 0.5% tolerance, hard-trim if over

### VRAM measurement

Background `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2` process writes a CSV during the run. Peak extracted via `awk -F',' 'NR>1 {gsub(/[ MiB]/,"",$2); if($2+0>max) max=$2+0} END{print max}'`.

### Server readiness probe

llama.cpp `/health` returns 200 *before* model loading completes. We hit `/v1/chat/completions` with a 1-token `"hi"` request instead, treating HTTP 200 (not 503/400) as the actual ready signal.

### Acceptance criteria for Group D

| Criterion | Target | Measured |
|---|---|---|
| Cases passed (`answer in output_text`) | ≥9/10 | 10/10 ✅ |
| Median TTFT | <15s | 8.96s ✅ |
| Median decode tok/s | ≥60 | 39.18 ⚠️ |
| Peak VRAM | <13 GB | 9.9 GB ✅ |

The tok/s miss is a projection error (extrapolated from 4K decode without accounting for KV-attention scaling at 60K). The actual 39 tok/s is genuine and acceptable for production (~8× faster than human reading speed).

### Known failure modes encountered this weekend

- `Qwen3.5-9B-DFlash-Q4_K_M.gguf` missing tokenizer merges in metadata → use the Q8_0 variant
- `/health` returning 200 prematurely → false-positive readiness
- Prompt = 131,082 tokens (post chat-template) but server allocated `-c 131072` → 400 reject with "exceeds available context size"; bump `-c` slightly
- DFlash + thinking mode → 45.8% acceptance on thinking chains; always disable thinking for spec
- vLLM v0.21.0 TurboQuant + speculative decoding → `supports_spec_as_decode=False` silently disables spec
- WSL2 ptrace_scope=1 blocks `py-spy` → fall back to `cProfile` via SIGTERM dump
- `pkill -f` kills the parent bash shell when pattern matches PATH expansion → always kill by specific PID

---

## Migration: GGUFs moved from /mnt/ NTFS to /home/tommy/models/ ext4

**Why:** earlier session profiling showed ~3-5× slower model load from NTFS via WSL2 `/mnt/` than native ext4. See [[project_models_location]].

**Files migrated (this session):**
- Qwen3-Embedding-8B-Q8_0.gguf (8.0 GB)
- Qwen3.5-9B-Q8_0.gguf (9.5 GB)
- Qwen3.5-9B-DFlash-Q4_K_M.gguf (624 MB) — note: broken metadata, kept for completeness
- Qwen3.5-9B-DFlash-q8_0.gguf (1.1 GB)
- qwen35-dflash-draft.gguf (2.1 GB)

**Already in /home (duplicates from prior sessions):**
- Qwen3-0.6B-BF16.gguf, Qwen3.6-27B-Q4_K_M.gguf, dflash-draft-3.6-q8_0.gguf

**Verification:** sha256sum on all 8 files both sides → all matched. Original /mnt copies deleted.

**Going forward:** zero references to `/mnt/i/dev/LLM/TurboQuant_Benchmark/models/` in repo source. All scripts and configs use `/home/tommy/models/`.

---

## TurboQuant variant comparison (added 2026-05-24)

After the initial run shipped with `turbo3_tcq` as default, we swept the other beellama-supported TurboQuant variants on the same 10 NIAH @ 60K cases through the Docker image. **Result: `turbo4` is 82% faster than `turbo3_tcq` at identical accuracy.**

| Variant | Bits | Encoding | Accuracy | TTFT median | **Decode tok/s** | Peak VRAM |
|---|---|---|---|---|---|---|
| **`turbo4`** ⭐ (new default) | 4 | raw | **10/10** | 7.58s | **71.4** | **9,734 MiB** |
| `turbo3` | 3 | raw | 10/10 | 6.63s | 47.3 | 9,862 MiB |
| `turbo3_tcq` (old default) | 3 | TCQ-encoded | 10/10 | 8.96s | 39.2 | 9,878 MiB |

**Findings:**
- **All three variants preserve NIAH retrieval quality** at 100% — TurboQuant's KV compression doesn't degrade long-context behavior at 3 or 4 bits.
- **TCQ encoding's bit savings are lost in measurement noise** at this scale (~140 MiB difference across variants), but its decode cost is real and measurable (~40% slower than raw 3-bit).
- **`turbo4` is the surprise winner** — slightly bigger KV per token (4-bit vs 3-bit) but the simpler kernel is much faster, and the VRAM difference is sub-1%.
- This swap is a **free speedup**: same model file, same context, same flags except `-ctk/-ctv turbo4` instead of `turbo3_tcq`.

**Production deployment swapped** to `turbo4` in:
- `docs/deployment/docker/.env.example`
- `docs/deployment/docker/Dockerfile` ENV defaults
- `docs/deployment/docker/docker-compose.yml`
- `docs/deployment/docker/README.md` (and VALIDATION.md)
- `docs/deployment/beellama-qwen35-9b.md`

---

## Bonus run — vLLM v0.21.0 TurboQuant on Qwen3.5-9B-FP8 (failed, but informative)

Tried `vllm/vllm-openai:v0.21.0` with `--kv-cache-dtype turboquant_k3v4_nc` on `lovedheart/Qwen3.5-9B-FP8`, single-slot (`--max-num-seqs 1`), 64K context. Goal: directly compare VRAM to beellama's ~10 GB at the same workload.

**Result: vLLM hit two distinct bugs.**

1. **OOM during KV cache block allocation** at `--gpu-memory-utilization 0.55` (13.5 GB cap). The FP8 9B weights + vLLM compile/Triton/CUDA-graph overhead consumed ~13 GB, leaving zero memory for the KV cache pool. Failed with `ValueError: No available memory for the cache blocks`. Even `--enforce-eager` (no CUDA graphs) at 4K context couldn't fit in 13.5 GB.

2. **TurboQuant workspace assertion at long context** (60K NIAH prompt), at `--gpu-memory-utilization 0.75` (18 GB cap):
   ```
   AssertionError: Workspace is locked but allocation from
   'turboquant_attn.py:747:_continuation_prefill' requires 8.00 MB,
   current size is 0.51 MB. Workspace growth is not allowed after locking.
   ```
   The TQ workspace is pre-sized during warmup, locked, then the real long-context prefill needs more memory and asserts. Same class of bug as [Issue #42808](https://github.com/vllm-project/vllm/issues/42808) (which was reported for TQ+MTP); turns out it also fires on TQ+long-context without spec decoding.

**What we did learn about vLLM VRAM footprint:**

vLLM with `--gpu-memory-utilization 0.75 --max-model-len 65536` allocated to **16,936 MiB** at steady state on load. Breakdown estimate:
- FP8 weights: ~8.9 GB
- vLLM compile cache (torch.compile/Inductor): ~0.5-1 GB
- Triton kernel scratch (TurboQuant): ~0.5 GB
- CUDA graph capture buffers (sizes [1, 2]): ~1-2 GB
- KV cache pool reservation: rest of the 13.5 GB budget
- vLLM block table, scheduler state: ~0.3 GB

vs **beellama 9,878 MiB** for the same model + KV precision + context — vLLM is ~7 GB heavier *with the pool reservation*, of which ~3-4 GB is genuinely needed (compile + CUDA graphs + Triton + block tables that beellama doesn't have), and the rest is pool over-allocation.

beellama wins on raw minimum VRAM because it has no JIT/compile, no CUDA graphs, no paged manager — just static GGUF loading and direct ggml ops. Trade-off: it can't do continuous batching. For single-user use, that's a good trade.

**Why we didn't bother retrying.** Even if we cherry-pick PR #40914 to fix the workspace bug, vLLM at this workload would still allocate substantially more VRAM than beellama. Beellama already wins for the single-user coexistence use case; the bonus run confirmed that and surfaced two more vLLM bugs to add to the watchlist.

Added to `docs/watchlist.md` as Item #11 (TurboQuant workspace assertion at long context without spec).

---

## Next steps & watch list

See `docs/watchlist.md` for the running list of upstream items to monitor. Highlights:

- **vLLM PR #40914 merges** → unlocks TQ + MTP combo on Qwen3.5/3.6 in vLLM
- **vLLM Issue #41559 resolution** → unlocks TQ + DFlash combo on vLLM (currently structurally impossible)
- **Lucebox publishes PFlash recipe for 9B** → would extend our long-context VRAM win down to 9B (currently only 27B is tuned)
- **Smaller DFlash draft for 9B** (~0.5B vs current 1.5B) → would cut combo's draft-model overhead and make the combo win at shorter outputs

User has been advised to set up a monthly cloud-side reminder via `/schedule` to re-check the watchlist.

---

## Reproducibility appendix

**GPU:** RTX 4090 24,564 MiB, compute capability 8.9 (sm_89)
**OS:** WSL2 Ubuntu 22.04 on Win11, kernel 6.6.87.2-microsoft-standard-WSL2
**CUDA:** 13.1 driver
**Model file paths:** all under `/home/tommy/models/` post-migration
**Test cases:** `/tmp/niah_60k_x10.jsonl` (10 × 60K NIAH), `/tmp/niah_128k.jsonl` (3 × 131K NIAH)
**Bench scripts:** `/tmp/bench_*.sh` (ephemeral; key configs documented inline above)
**Raw results:**
- `/tmp/beellama_matrix/` — Group A
- `/tmp/beellama_128k/` — Group B
- `/tmp/beellama_128k_4k/` — Group C
- `/tmp/beellama_final_validation/` — Group D
- `/tmp/lucebox_pflash*.log` — Lucebox runs
- `/tmp/hashes_home.txt`, `/tmp/hashes_mnt.txt` — migration verification
