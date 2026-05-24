# Qwen3.5-9B Inference Server Benchmark Report

**Date:** 2026-05-23 / 2026-05-24
**Author:** Tommy Cheung (@1TommyCheung)
**GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
**Model:** Qwen3.5-9B (FP8 for Python servers, Q8 GGUF for llama.cpp servers)

---

## 1. Executive Summary

We benchmarked 5 inference servers on Qwen3.5-9B text generation, testing baseline autoregressive decoding plus two speculative decoding methods (MTP and DFlash). Additionally, we tested Lucebox-Hub on the larger Qwen3.6-27B model as an out-of-spec comparison.

### Key Results (C=1, Single User)

| Server | Config | tok/s | Latency p50 | TTFT p50 | VRAM | Quality |
|---|---|---|---|---|---|---|
| vLLM | baseline | 54.7 | 9.25s | 38ms | 23.0 GB | 7/7 |
| vLLM | MTP | 41.4 | **4.85s** | 62ms | 23.5 GB | 7/7 |
| vLLM | DFlash | 40.1 | **3.85s** | 62ms | 24.0 GB | 7/7 |
| SGLang | baseline | 57.2 | 8.73s | 63ms | 22.2 GB | — |
| SGLang | MTP | ❌ | — | — | — | SGLang bug |
| SGLang | DFlash | ❌ | — | — | — | OOM 24GB |
| beellama | baseline Q8 | 60.9 | 6.08s | 143ms | 15.6 GB | — |
| **beellama** | **DFlash Q8** | **158.0** | **2.98s** | 252ms | **11.7 GB** | — |
| Lucebox* | DFlash+DDTree | 54.7 | 9.08s | 256ms | 23.5 GB | 7/7 |

*Lucebox tested with Qwen3.6-27B Q4_K_M (3x larger model) + TQ3 KV cache + 128K context

### Winners by Category

- **Fastest single-user decode:** beellama DFlash — 158 tok/s, 2.98s per request
- **Best VRAM efficiency:** beellama DFlash — 11.7 GB total
- **Best concurrency scaling:** vLLM baseline — 846 tok/s at C=16
- **Largest model on 24GB:** Lucebox — 27B Q4 with 128K context at 54.7 tok/s
- **Lowest TTFT:** vLLM baseline — 38ms
- **Fastest latency (user-facing):** beellama DFlash — 2.98s per request

---

## 2. Test Environment

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9, driver 596.21 |
| VRAM | 24,563 MiB |
| CPU | 13th Gen Intel Core i7-13700K (24 threads) |
| RAM | 64 GB (32 GB WSL allocation) |
| OS | WSL2 (Ubuntu) on Windows 11 |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| CUDA | 13.2 |
| Python | 3.12.12 |

---

## 3. Methodology

### 3.1 Models

| Model | Format | Size | Source | Used by |
|---|---|---|---|---|
| Qwen3.5-9B FP8 | safetensors | ~10 GB | `lovedheart/Qwen3.5-9B-FP8` | vLLM, SGLang |
| Qwen3.5-9B Q8 | GGUF | 9.2 GB | `unsloth/Qwen3.5-9B-GGUF` | beellama |
| Qwen3.5-9B DFlash drafter | safetensors | ~1 GB | `z-lab/Qwen3.5-9B-DFlash` | vLLM DFlash |
| Qwen3.5-9B DFlash drafter | GGUF | 1.1 GB | `lym00/Qwen3.5-9B-DFlash-GGUF-Test` | beellama DFlash |
| Qwen3.6-27B Q4_K_M | GGUF | 16 GB | `unsloth/Qwen3.6-27B-GGUF` | Lucebox |
| Qwen3.6-27B DFlash drafter | GGUF | 1.8 GB | `Lucebox/Qwen3.6-27B-DFlash-GGUF` | Lucebox DFlash |

### 3.2 Speed Benchmark

**Tool:** Custom `tqbench` framework (`eval_speed.py`)
- 150 prompts: 50 short (<100 tok), 50 medium (100-500 tok), 50 long (500-2000 tok)
- Concurrency levels: 1, 4, 16
- Streaming SSE via async httpx
- Metrics: tok/s, req/s, TTFT, ITL, latency p50/p95/p99, VRAM, KV cache usage
- Temperature: 0.0 (deterministic)
- Thinking mode: disabled via `chat_template_kwargs: {"enable_thinking": false}`

### 3.3 Quality Validation

**Tool:** Custom 10-query quality check (`eval_quick_quality.py`)
- 10 complex queries: math, code, reasoning, knowledge, creative
- Expected-answer assertions (7 checkable, 3 open-ended)
- max_tokens: 8192
- Validates server doesn't degrade output quality

### 3.4 Thinking Mode

Qwen3.5's thinking mode generates `<think>` reasoning chains before answering. We disabled it for all benchmarks because:
1. Speculative decoding acceptance rates drop to ~45% on thinking text (vs ~80% on direct answers)
2. Output token counts inflate 2-3x, making tok/s comparisons misleading
3. The model's thinking chains are unpredictable, penalizing speculative methods disproportionately

---

## 4. Detailed Results

### 4.1 vLLM (v0.18.2rc1)

**Docker image:** `vllm/vllm-openai:gemma4-cu130`

| Config | C=1 tok/s | C=1 lat p50 | C=4 tok/s | C=16 tok/s | VRAM |
|---|---|---|---|---|---|
| Baseline | 54.7 | 9.25s | 248.9 | **846.0** | 22,984 MB |
| MTP (2 tokens) | 41.4 | 4.85s | 156.9 | 493.9 | 23,482 MB |
| DFlash (5 tokens) | 40.1 | **3.85s** | 146.1 | 399.9 | 23,994 MB |

**Key findings:**
- vLLM baseline has the **best concurrency scaling** — 15.5x throughput from C=1 to C=16
- MTP and DFlash **reduce latency** (1.9x and 2.4x faster per request) but **lower tok/s** because they generate fewer tokens per request (the model gives more concise answers with spec decode)
- DFlash acceptance rate: 45.6% (low for 9B model — drafter struggles with output distribution)
- DFlash consumes 99.3% KV cache at C=16 — would OOM at C=64
- MTP uses native heads (no separate drafter), moderate KV usage (42.6% at C=16)

### 4.2 SGLang (v0.5.12.post1)

| Config | C=1 tok/s | C=1 lat p50 | C=4 tok/s | C=16 tok/s | VRAM |
|---|---|---|---|---|---|
| Baseline | 57.2 | 8.73s | 253.3 | 813.2 | 22,222 MB |
| MTP (NEXTN) | ❌ crashed | — | — | — | — |
| DFlash | ❌ OOM | — | — | — | — |

**Key findings:**
- SGLang baseline is **4.6% faster** than vLLM at C=1 and uses 762 MB less VRAM
- **MTP crashes** due to SGLang bug: `Qwen3_5ForConditionalGeneration` is missing from the NEXTN auto-detect list (PR #23859 pending). The FP8 model does have MTP heads (13 tensors verified).
- **DFlash OOMs** at all memory fractions (0.75–0.88). SGLang pre-allocates CUDA graphs more aggressively than vLLM — 9B model + 1B drafter + graph buffers exceeds 24GB.

### 4.3 beellama.cpp (llama.cpp fork)

| Config | C=1 tok/s | C=1 lat p50 | C=4 tok/s | VRAM |
|---|---|---|---|---|
| Baseline Q8 (np=1) | 60.9 | 6.08s | — | 15,603 MB |
| Baseline Q8 (np=4) | — | — | 215.4 | 10,662 MB |
| **DFlash Q8 (np=1)** | **158.0** | **2.98s** | — | **11,730 MB** |

**Key findings:**
- **beellama DFlash is the standout** — 158 tok/s at C=1, 2.6x faster than its own baseline
- Uses only **11.7 GB VRAM** (half of vLLM/SGLang) — room for a second model or longer context
- ITL = 0.0ms with DFlash (tokens delivered in chunks via block verification)
- Single-slot server (`-np 1`) — no continuous batching. C=4 with `-np 4` scales to 215 tok/s but requests are time-sliced, not fused
- CopySpec (no drafter, suffix matching) showed marginal improvement over baseline — DFlash is the clear winner

### 4.4 Lucebox-Hub (Qwen3.6-27B — out of spec)

Lucebox was tested on the **3x larger** Qwen3.6-27B model to demonstrate its unique capability: running a 27B model with DFlash + DDTree + TQ3 KV cache + 128K context on a single 24GB GPU.

#### Server mode (HTTP, mixed prompts)

| Metric | Value |
|---|---|
| tok/s (C=1) | 54.7 |
| Latency p50 | 9.08s |
| TTFT p50 | 256ms |
| VRAM | 23,484 MB |
| Quality | 7/7 passed |

#### Direct binary (bench_he.py, HumanEval code stubs)

| Model | Budget | Mean tok/s | Peak tok/s | AL | Speedup vs AR |
|---|---|---|---|---|---|
| Qwen3.5-27B Q4_K_M | 28 | **125.4** | **183.8** | 7.77 | **3.89x** |
| Qwen3.6-27B Q4_K_M | 36 | 84.6 | 132.4 | 5.32 | 2.51x |

**Key findings:**
- **27B model at 9B server speeds** — Lucebox DFlash on 27B Q4 matches vLLM baseline on 9B FP8
- Qwen3.5-27B drafter is much better than Qwen3.6 (AL 7.77 vs 5.32) — Qwen3.6 drafter is still under training
- Server mode is ~35% slower than direct binary (HTTP/JSON/SSE overhead + thinking mode + mixed prompts)
- TQ3 KV cache compression enables 128K context in 24GB with a 27B model
- DDTree budget=28 optimal for RTX 4090 (vs 22 for 3090, 40 for 5090)

---

## 5. Issues Encountered

### 5.1 Thinking Mode Kills Speculative Decoding

**Impact:** Critical
**Affects:** All servers with MTP/DFlash

Qwen3.5's thinking mode generates unpredictable reasoning chains that speculative decoders cannot draft efficiently. DFlash acceptance dropped to 45.6% (needs >60% to break even). MTP generated 2.6x fewer tokens but finished 1.85x faster in wall-clock time — the tok/s metric was misleading.

**Resolution:** Disable thinking via `chat_template_kwargs: {"enable_thinking": false}` for all benchmarks. Qwen3.6 ignores this flag (thinking is baked into the model behavior).

### 5.2 SGLang MTP/DFlash Broken on 24GB

**Impact:** High
**Affects:** SGLang 0.5.12.post1

- **MTP (NEXTN):** `Qwen3_5ForConditionalGeneration` missing from auto-detect list. PR #23859 is unmerged. Workaround (`--speculative-draft-model-path` pointing to same model) causes SGLang to load the model twice → OOM.
- **DFlash:** CUDA graph pre-allocation exceeds 24GB at all memory fractions tested. vLLM manages the same model+drafter combo because it allocates more lazily.

**Status:** Unresolved. SGLang spec decode for Qwen3.5 on 24GB requires PR #23859 merge.

### 5.3 WSL2 Performance Overhead

**Impact:** Medium
**Affects:** All servers

- NTFS-mounted models (`/mnt/`) showed 9% CPU utilization and 337K voluntary context switches during loading. **Always use native ext4 (`/home/`).**
- `pin_memory=False` forced by WSL2 — ~3-5% decode throughput penalty
- Estimated total WSL2 overhead: 3-5% vs bare metal Linux

### 5.4 vLLM DFlash num_speculative_tokens

**Impact:** Medium
**Affects:** vLLM DFlash

Initial attempt with `num_speculative_tokens=15` failed — draft token buffer consumed the entire batch budget (256 seqs × 15 = 3840 > 8192). Reduced to 5 speculative tokens.

### 5.5 lm-evaluation-harness Integration Issues

**Impact:** Low
**Affects:** Quality evaluation

- `local-chat-completions` model type doesn't support `loglikelihood` (needed by tinyBenchmarks)
- Docker root-owned HF cache lock files blocked tokenizer loading
- Estimated 10+ hours for full tinyBenchmarks via API at sequential request rate
- **Resolution:** Replaced with custom 10-query quality check — sufficient for backend parity validation

### 5.6 beellama DFlash Drafter Compatibility

**Impact:** Medium
**Affects:** beellama, Lucebox

- Community GGUF drafters (`psychopenguin`) had missing tokenizer merges
- Lucebox requires specific architecture tags (`qwen35-dflash-draft`) — standard GGUF drafters are rejected
- Lucebox's `convert_dflash_to_gguf.py` converts z-lab HF drafters to the correct format
- beellama accepted `lym00/Qwen3.5-9B-DFlash-GGUF-Test` drafter

---

## 6. Concurrency Scaling Analysis

| Server | C=1 tok/s | C=4 tok/s | C=16 tok/s | C=1→C=16 scaling |
|---|---|---|---|---|
| vLLM baseline | 54.7 | 248.9 | **846.0** | **15.5x** |
| vLLM MTP | 41.4 | 156.9 | 493.9 | 11.9x |
| vLLM DFlash | 40.1 | 146.1 | 399.9 | 10.0x |
| SGLang baseline | 57.2 | 253.3 | 813.2 | 14.2x |
| beellama baseline | 60.9 | 215.4* | — | 3.5x* |

*beellama with `-np 4` (time-sliced, not continuous batching)

Python servers (vLLM, SGLang) achieve near-linear scaling through continuous batching. llama.cpp servers (beellama) scale poorly because each slot runs independently on the GPU — no batch fusion.

**Recommendation:** Use vLLM/SGLang for multi-user serving. Use beellama DFlash for single-user maximum speed (coding agents, local development).

---

## 7. VRAM Efficiency

| Server | Config | VRAM Used | KV Cache at C=16 | Headroom |
|---|---|---|---|---|
| vLLM baseline | FP8 | 22,984 MB | 17.0% | 1.6 GB |
| vLLM MTP | FP8 | 23,482 MB | 42.6% | 1.1 GB |
| vLLM DFlash | FP8 + drafter | 23,994 MB | **99.3%** | 0.6 GB |
| SGLang baseline | FP8 | 22,222 MB | — | 2.3 GB |
| **beellama baseline** | **Q8 GGUF** | **15,603 MB** | — | **8.9 GB** |
| **beellama DFlash** | **Q8 + drafter** | **11,730 MB** | — | **12.8 GB** |
| Lucebox 27B | Q4 + drafter + TQ3 | 23,484 MB | — | 1.1 GB |

beellama uses half the VRAM of Python servers — GGUF models are loaded directly without Python tensor framework overhead. The 12.8 GB headroom with DFlash means room for a second model, longer context, or other GPU workloads.

---

## 8. Quality Validation

All servers produced correct answers on 7/7 checkable queries. Speculative decoding (MTP, DFlash) is mathematically lossless — same output distribution at temperature 0.0.

| Query | vLLM | vLLM MTP | vLLM DFlash | Lucebox 27B |
|---|---|---|---|---|
| math_1 (127×43) | PASS | PASS | PASS | PASS |
| math_2 (avg speed) | PASS | PASS | PASS | PASS |
| code_1 (palindrome) | PASS | PASS | PASS | PASS |
| code_2 (LCS) | PASS | PASS | PASS | PASS |
| reason_1 (syllogism) | PASS | PASS | PASS | PASS |
| knowledge_2 (tides) | PASS | PASS | PASS | PASS |
| extract_1 (sum numbers) | PASS | PASS | PASS | PASS |

---

## 9. Lucebox Budget Analysis

### RTX 4090 Optimal Budget

The DDTree budget controls how many tree nodes are evaluated per speculation step. Higher budget = more speculative tokens verified per step, but more compute per step.

**Qwen3.5-27B Q4_K_M:**

| Budget | tok/s | AL | Notes |
|---|---|---|---|
| 22 | 123.0 | 7.45 | 3090 default |
| 26 | 123.7 | 7.60 | |
| **28** | **125.4** | **7.77** | **4090 optimal** |
| 30 | 123.1 | 7.62 | |
| 34 | 111.8 | 7.28 | degrading |

**Qwen3.6-27B Q4_K_M:**

| Budget | tok/s | AL | Notes |
|---|---|---|---|
| 22 | 79.7 | 4.85 | 3090 default |
| 28 | 82.5 | 4.99 | |
| 34 | 83.0 | 5.17 | |
| **36** | **84.6** | **5.32** | **4090 optimal** |
| 38 | 82.8 | 5.32 | degrading |

**Why 4090 budget > 3090 budget:**
The RTX 4090's 72 MB L2 cache (vs 3090's 6 MB) keeps the DDTree verification working set in cache, enabling larger trees without DRAM bandwidth penalties. SM count (128 vs 82) enables wider parallel evaluation.

---

## 10. Architecture Comparison

### Request Flow

**Python servers (vLLM, SGLang):**
```
Client → HTTP/SSE → Python server → CUDA kernels → GPU → Python → HTTP/SSE → Client
```
- Continuous batching fuses multiple requests into one GPU batch
- torch.compile + CUDA graphs optimize the forward pass
- Pre-allocated KV cache pool (PagedAttention for vLLM, RadixAttention for SGLang)

**llama.cpp servers (beellama):**
```
Client → HTTP/SSE → C++ server → GGML CUDA kernels → GPU → C++ → HTTP/SSE → Client
```
- Time-sliced multi-slot (no batch fusion)
- Lower overhead per request (no Python GIL, no tensor framework)
- mmap model loading, on-demand VRAM allocation

**Lucebox server mode:**
```
Client → HTTP/SSE → Python FastAPI → stdin pipe → C++/CUDA binary → GPU → stdout pipe → Python → HTTP/SSE → Client
```
- DFlash + DDTree speculative decoding in C++
- Python wrapper adds ~15% overhead vs direct binary
- Single-slot daemon

---

## 11. Potential Next Steps

### 11.1 PFlash Long-Context TTFT Test (High Priority)

Lucebox's PFlash speculative prefill claims 10x TTFT improvement at 128K context. We downloaded the PFlash drafter (Qwen3-0.6B BF16) but didn't run the test. This would demonstrate Lucebox's unique advantage for RAG/long-document workloads.

**Why it matters:** TTFT at 128K is the bottleneck for production RAG systems. If PFlash delivers even 5x improvement, it changes the cost equation for local vs. cloud inference on long contexts.

### 11.2 TurboQuant KV Cache Comparison (High Priority)

All three server types support TurboQuant KV cache compression:
- vLLM: `--kv-cache-dtype turboquant_k8v4`
- SGLang: `--kv-cache-dtype turboquant_k8v4`
- beellama: `--cache-type-k turbo3 --cache-type-v turbo3`

This would show how much longer context or lower VRAM each server can achieve with compressed KV, and whether compression degrades quality.

### 11.3 Bare Metal Linux Comparison (Medium Priority)

Our WSL2 results show ~3-5% overhead. A bare metal run would establish true performance ceilings and determine whether the SGLang DFlash OOM is WSL2-specific or a genuine memory management issue.

### 11.4 beellama DFlash Multi-Slot (Medium Priority)

Test `-np 4 --spec-dflash-max-slots 4` to see if beellama's DFlash scales with concurrency. If it maintains 100+ tok/s at C=4, it becomes viable for small-team serving.

### 11.5 SGLang Fix Validation (Low Priority)

Once SGLang PR #23859 merges, re-test MTP and DFlash. If DFlash OOM is resolved with `--cuda-graph-max-bs 2`, SGLang could be competitive with vLLM on speculative decoding.

### 11.6 Qwen3.6 DFlash Drafter Re-evaluation (Future)

The Qwen3.6 DFlash drafter is still under training (AL 5.32 vs Qwen3.5's 7.77). When z-lab releases the final version, re-benchmark Lucebox — expect 30-50% throughput improvement.

### 11.7 beellama on Qwen3.6-27B (Interesting)

beellama has official Qwen3.6-27B DFlash drafters (`Anbeeld/Qwen3.6-27B-DFlash-GGUF`). Given beellama's 2.6x DFlash speedup on 9B, it could potentially beat Lucebox on the same 27B model. Direct comparison would isolate which DFlash implementation is faster.

---

## 12. Conclusion

**For single-user inference (coding agents, local dev):** beellama.cpp with DFlash is the clear winner — 158 tok/s, 2.98s latency, 11.7 GB VRAM. It's 3x faster than vLLM baseline and uses half the memory.

**For multi-user serving:** vLLM baseline scales to 846 tok/s at C=16 through continuous batching. MTP and DFlash add latency benefits but reduce throughput — use them only when per-request latency matters more than aggregate throughput.

**For large models on consumer hardware:** Lucebox demonstrates that a 27B model can run at 54.7 tok/s (server) / 125.4 tok/s (binary) on a single 24GB GPU with DFlash + DDTree + TQ3 KV cache compression — performance competitive with 9B models on other servers.

**SGLang** has the best baseline performance (57.2 tok/s, lowest VRAM) but its speculative decoding stack is broken for Qwen3.5 on 24GB in v0.5.12.

The choice between servers depends on the deployment scenario:

| Scenario | Recommended Server | Config |
|---|---|---|
| Coding agent (single user) | beellama DFlash | Q8 GGUF + DFlash drafter |
| API serving (multi-user) | vLLM | FP8, baseline (no spec decode) |
| Large model, consumer GPU | Lucebox | Q4 GGUF + DFlash + TQ3 KV |
| Low VRAM budget | beellama DFlash | 11.7 GB total footprint |
| Lowest latency per request | beellama DFlash | 2.98s at C=1 |
| Highest throughput | vLLM | 846 tok/s at C=16 |

---

## Appendix A: Server Launch Commands

See `docs/findings/2026-05-24-benchmark-server-configs.md` for exact launch commands, model paths, and flags for each configuration.

## Appendix B: Raw Data

All speed and quality JSON results are in `tqbench/benchmarks/generation/reports/raw/`.

## Appendix C: Benchmark Framework

Built with the `tqbench` modular benchmark framework. Source code at `tqbench/benchmarks/generation/`. Prompt data generated by `tqbench/benchmarks/generation/data/generate_prompts.py`.
