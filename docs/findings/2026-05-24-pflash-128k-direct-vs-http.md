# PFlash @ 128K: Direct binary vs HTTP server.py + Python wrapper overhead

**Date:** 2026-05-24
**Hardware:** RTX 4090 24GB, WSL2 (Ubuntu 22.04, host Win11)
**Binary:** `lucebox-hub/dflash/build/test_dflash` compiled for sm_89 only (verified via `cuobjdump`)

## Direct binary @ 128K NIAH, n=3

Same setup as Lucebox's reference: Qwen3.6-27B Q4_K_M target + Qwen3-0.6B BF16 drafter, `--bsa 1 --alpha 0.85 --keep-ratio 0.05 --ddtree-budget 36 --kv-tq3 1 --no-thinking`. Source 131,068 → kept 6,524 (20.1×).

| Case | TTFT | Drafter forward (A_compute + FP) | Tail score | Accuracy |
|---|---|---|---|---|
| 0 (cold) | 156.5s | 117.6s (31.6s + 21.7s) | 28.6s | ✅ |
| 1 (warm) | 154.1s | 117.8s (32.6s + 34.8s) | 4.3s | ✅ |
| 2 (warm) | 176.3s | 115.4s (31.0s + 22.0s) | 30.1s | ✅ |

**Steady-state ≈ cold.** Warmup is NOT the cause of the gap to Lucebox's 24.8s claim.

## Comparison vs Lucebox README claim

| Setup | TTFT @ 128K | GPU |
|---|---|---|
| Lucebox README claim | **24.8s** | RTX 3090 (sm_86, native Linux) |
| Our direct binary | **~160s** | RTX 4090 (sm_89, WSL2) |
| **Gap** | **~6.5×** | |

## Falsified hypotheses

- **sm_86-tuned kernels with sm_89 JIT fallback** — false. `cuobjdump` shows every kernel compiled native sm_89 SASS (`CMAKE_CUDA_ARCHITECTURES=89`). 4090 has more BF16 TFLOPS (165 vs 71) and memory bandwidth (1008 vs 936 GB/s) than 3090, so should be faster, not slower.
- **CUDA JIT / kernel cache warmup on first run** — false. n=3 shows cases 1 and 2 within ±14% of case 0.
- **BSA not enabled** — false. Confirmed `--bsa 1` and `DFLASH_FP_USE_BSA=1` env var on every run.
- **Wrong drafter precision (FP8/Q8 hypothesis)** — false. Lucebox `pflash/README.md` explicitly mandates BF16 Qwen3-0.6B drafter; we use exactly that GGUF.
- **Kernel-launch overhead from WSL2** — false. ~60K launches × 50µs would only be ~3s, not 130s+.

## Remaining real candidates (not yet bisected)

1. **WSL2 GPU memory bandwidth penalty** under sustained BSA workload. Plausible: 20–30% off documented in past WSL2 issues. Would only account for ~30% gap, not 6×.
2. **GPU power state ceiling under WSL2.** GPU idles at 210 MHz vs 3120 MHz max. WSL2 has known cases where the driver doesn't aggressively boost during memory-bound work. Couldn't lock clocks (sudo password not available); locked-clocks test pending.
3. **`tok_embd 682 MiB CPU-only (q4_K)`** for the 0.6B drafter. CPU dequant + DMA over 131K tokens on WSL2 VHD is brutal. Cannot be tested without modifying the binary to GPU-offload the embedding.

## HTTP vs direct binary: previous claim retracted

Earlier in the session I claimed "HTTP server.py @ 128K = 67s, direct binary @ 128K = 156s." **The 67s number was NOT at 128K.** Re-reading `/tmp/lucebox_pflash2.log`:

| Path | Context (source tokens) | Total wall | Compress phase |
|---|---|---|---|
| HTTP server.py | **87K** | 66.2s | 21.9s |
| Direct binary | **128K** | 156s | 152s |

These are different context sizes; the "HTTP is faster" conclusion was bogus. **No 128K HTTP measurement exists in our logs.** Scaling 21.9s @ 87K → 128K via O(S²) attention ≈ 47s estimate; via O(S log S) sparse ≈ 35s — neither matches the direct-binary's 152s. This suggests `bench_niah_cpp.py` and `server.py` may hit the daemon via different code paths (env vars, daemon stdin command sequence, or prefill cache state). Worth one apples-to-apples HTTP @ 128K run before drawing further conclusions.

## Python wrapper overhead — server.py profiling

`py-spy` blocked by WSL2 `ptrace_scope=1`. Used cProfile via SIGTERM-dump wrapper instead. 5 streaming requests, 100 in / 256 out tokens, daemon at 61–74 tok/s.

| Bucket | Time/request | % wall |
|---|---|---|
| `posix.read` blocking on daemon stdout pipe | ~4.09s | **97%** |
| Python CPU (tokenize, SSE, JSON, asyncio) | ~0.11s | **3%** |

Cross-check: 1290 `posix.read` × 16.2ms = exactly matches 63–70 tok/s daemon output rate.

**Top 3 by self-time:** `posix.read` (20.7s blocking on daemon), `uvicorn.serve` (9.2s asyncio epoll idle), `starlette.responses.wrap` (4.2s SSE generator — includes await time).

`json.dumps`: 36 calls total, all on startup/non-streaming. `re`: absent from top-40 during inference.

**Startup-only cost:** `gguf_reader._get_field_parts` 25.4s reading 27B GGUF metadata twice (`_arch_from_gguf` + `_tokenizer_id_from_gguf`). One-time at boot.

## Rust/Go rewrite verdict

**Not worthwhile.** ~3% of latency is Python. A perfect zero-overhead Rust replacement saves ~0.11s on a 4.09s request — **2.7%, inside noise**. TechEmpower's 30–50× Rust/Python throughput edge is irrelevant when the wrapper is I/O-blocked on a subprocess.

Legitimate optimization targets (in priority order):
1. Daemon-side: acceptance rate, CUDA kernel tuning, async prefill
2. First-request prefill: Rust `tokenizers` could shave ~50ms on chat template — already mostly eliminated by prefix cache
3. Could swap to Lucebox's bundled C++ `dflash_server` instead of `server.py` for ~3% savings if it ever matters

Retract earlier "15–35% Python overhead" estimate — measured value is **~3%**.
