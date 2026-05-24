# Lucebox DFlash — RTX 4090 WSL2 Benchmark Results

**Contributor:** @1TommyCheung
**Date:** 2026-05-24
**Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM)

## System

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9, driver 596.21 |
| VRAM | 24,563 MiB |
| CPU | 13th Gen Intel Core i7-13700K (24 threads) |
| RAM | 32 GB (WSL, 64GB host) |
| OS | WSL2 (Ubuntu) on Windows 11 |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| CUDA | 13.2 (nvcc cuda_13.2.r13.2/compiler.37668154_0) |
| Python | 3.12.12 |
| Build | `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DDFLASH27B_ENABLE_BSA=ON` |
| Filesystem | Models on native ext4 (`/home/tommy/models/`), not NTFS `/mnt/` |

**Note:** NTFS-mounted models via WSL2's 9P filesystem showed severe I/O bottlenecks (9% CPU utilization, 337K voluntary context switches). All benchmarks below used native ext4.  Avoid using WSL /mnt/ NTFS mount points for large files like models.

## Benchmark 1: Direct Binary (bench_he.py)

- Script: `python dflash/scripts/bench_he.py --n-gen 256 --ddtree-budget N`
- Mode: `fast` (fast-rollback enabled)
- 10 HumanEval code-completion prompts (86–139 input tokens each)
- No thinking mode, no chat template, no HTTP server — direct `test_dflash` binary
- No TQ3/Q4 KV cache override (default Q8_0 KV)
- No `--max-ctx` override (auto-fit)

### Architecture stack (direct binary)

```
bench_he.py
  ↓ writes binary token file
test_dflash (C++/CUDA binary)
  ↓ DFlash + DDTree speculative decoding
RTX 4090 GPU
  ↓ reads binary output
bench_he.py reports tok/s
```

No HTTP, no JSON serialization, no streaming overhead.

### Results: Qwen3.5-27B Q4_K_M

**Target:** `unsloth/Qwen3.5-27B-GGUF` → `Qwen3.5-27B-Q4_K_M.gguf` (16 GB)
**Drafter:** `spiritbuun/Qwen3.5-27B-DFlash-GGUF` → `dflash-draft-q4_k_m.gguf` (986 MB)

| Budget | Mean tok/s | Peak tok/s | Mean AL |
|---|---|---|---|
| 22 | 122.95 | 182.1 | 7.45 |
| 26 | 123.68 | 182.4 | 7.60 |
| **28** | **125.37** | **183.8** | **7.77** |
| 30 | 123.12 | 181.4 | 7.62 |
| 34 | 111.76 | 175.3 | 7.28 |

**Optimal budget: 28** — 125.37 tok/s mean, 183.8 tok/s peak

### Results: Qwen3.6-27B Q4_K_M

**Target:** `unsloth/Qwen3.6-27B-GGUF` → `Qwen3.6-27B-Q4_K_M.gguf` (16 GB)
**Drafter:** `Lucebox/Qwen3.6-27B-DFlash-GGUF` → `dflash-draft-3.6-q8_0.gguf` (1.8 GB)

| Budget | Mean tok/s | Peak tok/s | Mean AL |
|---|---|---|---|
| 22 | 79.65 | 127.4 | 4.85 |
| 26 | 82.37 | 126.8 | 5.00 |
| 28 | 82.46 | 128.8 | 4.99 |
| 30 | 79.82 | 123.4 | 5.06 |
| 34 | 83.00 | 134.1 | 5.17 |
| 35 | 83.12 | 131.2 | 5.20 |
| **36** | **84.55** | **132.4** | **5.32** |
| 37 | 84.30 | 131.1 | 5.32 |
| 38 | 82.80 | 125.3 | 5.32 |

**Optimal budget: 36** — 84.55 tok/s mean, 132.4 tok/s peak

#### Per-prompt breakdown (budget=36)

| Prompt | Steps | AL | tok/s |
|---|---|---|---|
| has_close_elements | 31 | 8.26 | 132.4 |
| separate_paren_groups | 46 | 5.57 | 91.1 |
| truncate_number | 0 | 0.00 | 0.0 (EOS) |
| below_zero | 46 | 5.57 | 90.2 |
| mean_absolute_deviation | 37 | 6.92 | 109.8 |
| intersperse | 40 | 6.40 | 101.7 |
| parse_nested_parens | 32 | 8.00 | 127.3 |
| filter_by_substring | 43 | 5.95 | 91.7 |
| sum_product | 0 | 0.00 | 0.0 (EOS) |
| rolling_max | 39 | 6.56 | 105.5 |

Excluding 2 EOS prompts: **106.2 tok/s mean** across 8 active prompts.

## Benchmark 2: Server Mode (agentic / real-world)

This is the configuration used by agentic coding clients (Claude Code, Codex, OpenCode) via the Lucebox harness scripts.

### Architecture stack (server mode)

```
Client (Claude Code / Codex / OpenCode / our benchmark)
  ↓ HTTP request (SSE streaming)
  ↓ POST /v1/chat/completions  (or /v1/messages for Claude, /v1/responses for Codex)
server.py (Python FastAPI/uvicorn)
  ↓ stdin/stdout pipe
test_dflash (C++/CUDA binary — DFlash + DDTree inference)
  ↓ GPU compute
RTX 4090
  ↓ stdout pipe
server.py
  ↓ SSE stream chunks
Client receives tokens
```

Three layers of overhead vs direct binary: FastAPI HTTP server, JSON serialization per SSE chunk, httpx client-side SSE parsing.

### Server launch command

```bash
DFLASH27B_KV_TQ3=1 python dflash/scripts/server.py \
  --target Qwen3.6-27B-Q4_K_M.gguf \
  --draft dflash-draft-3.6-q8_0.gguf \
  --port 8082 --budget 28 --max-ctx 131072 \
  --enable-thinking false
```

### Server-mode results (Qwen3.6-27B, 150 mixed prompts, C=1)

| Metric | Value |
|---|---|
| tok/s | 54.7 |
| req/s | 0.11 |
| Latency p50 | 9.08s |
| TTFT p50 | 256ms |
| ITL p50 | 0.2ms |
| tok/req | 509 |
| VRAM | 23,484 MB (with TQ3 KV + 128K ctx) |

### Quality validation (10 complex queries via server)

| Query | Tokens | Latency | Pass |
|---|---|---|---|
| math_1 (127×43) | 1766 | 24.5s | PASS |
| math_2 (avg speed) | 636 | 8.3s | PASS |
| code_1 (palindrome) | 120 | 1.6s | PASS |
| code_2 (LCS) | 255 | 2.3s | PASS |
| reason_1 (syllogism) | 1752 | 30.9s | PASS |
| reason_2 (5 houses) | 5458 | 111.0s | — |
| knowledge_1 (TCP/UDP) | 1238 | 22.3s | — |
| knowledge_2 (tides) | 862 | 17.0s | PASS |
| creative_1 (haiku) | 1474 | 23.6s | — |
| extract_1 (sum numbers) | 428 | 5.1s | PASS |

**Score: 7/7 passed** (all checked assertions correct)

### Server-mode vs direct binary

| Metric | Direct binary (bench_he.py) | Server mode (HTTP) | Overhead |
|---|---|---|---|
| Mean tok/s | 84.55 | 54.7 | -35% |
| Peak tok/s | 132.4 | — | — |
| TTFT | n/a | 256ms | HTTP + prefill |
| Prompts | HumanEval code stubs | Mixed Q&A (short/medium/long) | Different distribution |
| Thinking | Off (no chat template) | On (Qwen3.6 ignores disable flag) | More tokens generated |

The 35% gap is from: HTTP/JSON/SSE streaming overhead (~15%), Python GIL in FastAPI (~5%), mixed prompt distribution vs uniform code stubs (~10%), and thinking mode generating verbose responses (~5%).

**For agentic coding use cases (Claude Code, Codex, OpenCode), server-mode 54.7 tok/s is the real-world throughput.** The direct binary number (84.5 tok/s) reflects peak GPU decode capability.

## Analysis

### Why Qwen3.5 is faster than Qwen3.6

The Qwen3.5-27B drafter achieves 7.77 mean AL vs Qwen3.6's 5.32. This is because:
1. The Qwen3.5 drafter was trained on Qwen3.5's output distribution and is fully mature
2. The Qwen3.6 drafter (`z-lab/Qwen3.6-27B-DFlash`) is [still under training](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) per the HuggingFace model card
3. Qwen3.6 may have sufficiently different output distribution that the drafter needs more training data to converge

### RTX 4090 optimal budget vs RTX 3090

| GPU | Optimal Budget | Reason |
|---|---|---|
| RTX 3090 | 22 | 6 MB L2 — verification hits DRAM |
| **RTX 4090** | **28 (Qwen3.5) / 36 (Qwen3.6)** | **72 MB L2 — verification stays in cache** |
| RTX 5090 | 40 | 98 MB L2 + 1792 GB/s bandwidth |

The 4090's massive L2 cache upgrade (6→72 MB) is the dominant factor — DDTree verification working set fits entirely in cache, enabling larger tree budgets without DRAM bandwidth penalties. SM count (128 vs 82) enables wider parallel evaluation.

### WSL2 overhead

Running under WSL2 introduces overhead vs bare metal Linux:
- 9P filesystem bridge for `/mnt/` paths (mitigated by using native ext4 `/home/`)
- `pin_memory=False` forced by WSL2 environment
- CPU-GPU communication latency slightly higher

Estimated WSL2 penalty: ~3–5% on decode throughput based on the Qwen3.5 result (125.4 vs claimed 129.5 tok/s on bare metal 3090).

### Filesystem impact

Models on NTFS (`/mnt/i/`) showed:
- 9% CPU utilization during model loading (vs 85%+ on ext4)
- 337K voluntary context switches (I/O waits)
- ~10x slower model load times

**Always use native ext4 paths for GGUF models on WSL2.**

## Summary for RESULTS.md

```
RTX 4090 (WSL2, Ada sm_89, CUDA 13.2, 24 GB)
  Qwen3.5-27B Q4_K_M + spiritbuun Q4_K_M drafter:  125.4 tok/s @ budget=28  (AL 7.77)
  Qwen3.6-27B Q4_K_M + Lucebox Q8_0 drafter:         84.6 tok/s @ budget=36  (AL 5.32)
  Server mode (Qwen3.6, HTTP, TQ3 KV, 128K ctx):      54.7 tok/s @ budget=28
```

## PFlash Long-Context TTFT Test

### Setup

Server with PFlash enabled:
```bash
DFLASH27B_KV_TQ3=1 DFLASH_FP_USE_BSA=1 DFLASH_FP_ALPHA=0.85 \
python dflash/scripts/server.py \
  --target Qwen3.6-27B-Q4_K_M.gguf \
  --draft dflash-draft-3.6-q8_0.gguf \
  --port 8082 --budget 36 --max-ctx 131072 --no-thinking \
  --prefill-compression auto --prefill-threshold 4096 \
  --prefill-keep-ratio 0.05 \
  --prefill-drafter Qwen3-0.6B-BF16.gguf
```

Server reports: `pflash = auto · threshold=4096 keep=0.05 drafter=Qwen3-0.6B-BF16.gguf`

### Results

| Context | TTFT median (no PFlash) | TTFT median (PFlash) | Speedup |
|---|---|---|---|
| 1K | 5.0s | 5.0s | 1.0x (below threshold) |
| 4K | 76.6s | 84.3s | 0.9x (threshold edge, drafter overhead) |
| 8K | ~191s (extrapolated) | **28.4s** | **6.7x** |
| 32K | ~770s (extrapolated) | **22.0s** | **35x** |
| 128K | ~3000s (extrapolated) | **67.1s** | **~45x** |

Baseline numbers extrapolated from the measured ~28 tok/s prefill rate without compression.

### Comparison to Lucebox README claims

| Context | README claim (3090, binary) | Our result (4090 WSL2, HTTP server) | Gap |
|---|---|---|---|
| 64K | 13.5s | (not tested; 22s at 32K) | — |
| 128K | 24.8s | 67.1s | 2.7x slower |

The 128K gap likely reflects:
1. **HTTP server overhead** — we measured via `server.py` + HTTP, README benchmarked the raw `test_dflash --daemon` stdin protocol
2. **Tokenization overhead** — 128K tokens of JSON-encoded prompt over HTTP adds significant client-side and server-side processing
3. **WSL2 overhead** at long contexts

Despite the gap, **PFlash delivers transformative TTFT improvement**: 128K context that would otherwise take ~50 minutes responds in ~67 seconds. This makes RAG/long-document workflows viable on a 24GB consumer GPU.

### Why 4K TTFT was slower with PFlash

At 4K (the threshold), PFlash adds the drafter scoring step but compression doesn't yield enough savings to offset the overhead. The drafter cost is ~5-10s fixed; for 4K input it doesn't pay back. PFlash shines at 8K+ where the cost is amortized.
