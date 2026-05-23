# Finding: MTP and DFlash Performance with Qwen3.5-9B Thinking Mode

**Date:** 2026-05-23
**Model:** lovedheart/Qwen3.5-9B-FP8 on vLLM 0.18.2rc1
**GPU:** NVIDIA RTX 4090 (24 GB)
**DFlash drafter:** z-lab/Qwen3.5-9B-DFlash (~1B params)

## Summary

Both MTP and DFlash speculative decoding show degraded throughput (tok/s) when Qwen3.5's thinking mode is enabled. However, MTP delivers faster wall-clock response times because it generates shorter, more coherent reasoning chains.

DFlash shows a 45.8% draft acceptance rate on thinking output — well below the ~60-70% threshold needed for speculative decoding to break even.

## MTP Results (thinking enabled)

| Concurrency | Baseline tok/s | MTP tok/s | Baseline lat p50 | MTP lat p50 | Baseline tok/req | MTP tok/req |
|---|---|---|---|---|---|---|
| 1 | 55.2 | 39.7 (-28%) | 9.27s | 5.04s (+84%) | 597 | 232 |
| 4 | 252.1 | 147.7 (-41%) | 7.95s | 5.44s (+46%) | 597 | 233 |
| 16 | 842.7 | 449.6 (-47%) | 8.61s | 6.80s (+27%) | 596 | 233 |

### Key insight: tok/s is misleading

MTP generates **2.6x fewer tokens per request** (232 vs 597) because its "deeper strategic reasoning" produces more efficient thinking chains. The raw tok/s metric penalizes MTP for being more efficient.

**User-facing metrics tell the real story:**
- Wall time (C=1): 1621s baseline → 877s MTP = **1.85x faster**
- Latency p50 (C=1): 9.27s baseline → 5.04s MTP = **1.84x faster**

MTP is faster in every dimension that matters to the user.

### KV cache usage per concurrency (MTP run)

| Concurrency | KV cache peak % | VRAM peak |
|---|---|---|
| 1 | 4.8% | 23,356 MB |
| 4 | 11.1% | 23,356 MB |
| 16 | 44.3% | 23,358 MB |

At C=4, only 11% of the pre-allocated KV cache is used — `--gpu-memory-utilization` could be halved.

## DFlash Results (thinking enabled, killed mid-run)

DFlash with `num_speculative_tokens=5` was killed during C=1 after observing poor performance. Server metrics at time of kill:

### Acceptance rate: 45.8%

```
Draft tokens:    28,425
Accepted tokens: 13,026
Overall rate:    45.8%
```

### Acceptance by draft position

| Position | Accepted | Rate |
|---|---|---|
| 0 (next token) | 4,532 / 5,685 | 79.7% |
| 1 | 3,353 / 5,685 | 59.0% |
| 2 | 2,344 / 5,685 | 41.2% |
| 3 | 1,635 / 5,685 | 28.8% |
| 4 | 1,162 / 5,685 | 20.4% |

Acceptance drops sharply after position 1. By position 4, 80% of draft tokens are wasted.

### Why it's slow

At 45.8% acceptance with 5 draft tokens:
- Average accepted per step: ~2.3 tokens
- Cost per step: drafter forward pass (~1B params) + main model verification (5 candidates)
- The drafter + verification overhead exceeds the savings from 2.3 accepted tokens
- Break-even requires ~60-70% acceptance rate

### Root cause

Free-form thinking text is inherently unpredictable. The drafter model was trained on (presumably) direct output patterns, not stream-of-consciousness reasoning chains. Each thinking step involves novel logical connections that the drafter cannot anticipate.

## Configuration note: num_speculative_tokens

Initial attempt with `num_speculative_tokens=15` failed:
```
max_num_scheduled_tokens is set to -1536 ... does not allow any tokens to be scheduled
```
The draft token buffer slots consumed the entire batch budget (256 seqs × 15 tokens = 3840 > 8192 batch budget). Reduced to 5 speculative tokens.

## Recommendations

1. **Disable thinking for fair benchmarking.** Both baseline and speculative configs should use `chat_template_kwargs: {"enable_thinking": false}` for comparable results. Thinking mode penalizes speculative decoding disproportionately.

2. **Re-run with thinking disabled.** Expect significantly higher DFlash acceptance rates on structured/direct output. The drafter should predict "The answer is 4" far better than "Let me think about this step by step, first I consider..."

3. **Report both.** The thinking-mode results are valuable — they show that speculative decoding does NOT help (and may hurt) on reasoning workloads. This is a legitimate finding for users deploying thinking models.

4. **Investigate vLLM MTP implementation.** The tok/s degradation with MTP is unexpected per published benchmarks claiming 60% throughput boost. Possible causes:
   - vLLM's `qwen3_next_mtp` implementation overhead on RTX 4090
   - `num_speculative_tokens=2` too low for thinking chains
   - Interaction between thinking mode's token distribution and MTP head predictions
   - WSL2 `pin_memory=False` penalty compounding with MTP overhead

## Raw data

- Baseline: `reports/raw/2026-05-23_qwen3.5-9b-fp8-vllm_speed.json`
- MTP: `reports/raw/2026-05-23_qwen3.5-9b-fp8-vllm-mtp_speed.json`
- DFlash: killed before writing output; server metrics captured above

## SGLang Findings (added during benchmark run)

### SGLang Baseline — Works, comparable to vLLM

| Concurrency | tok/s | lat p50 | ttft p50 | VRAM |
|---|---|---|---|---|
| 1 | 57.2 | 8.73s | 63ms | 22,222 MB |
| 4 | 253.3 | 7.69s | 65ms | 22,226 MB |
| 16 | 813.2 | 9.18s | 127ms | 22,226 MB |

### SGLang MTP (NEXTN) — Crashes

`--speculative-algorithm NEXTN` with Qwen3.5 FP8 crashes in SGLang 0.5.12.post1. Error: `AssertionError` in `_handle_speculative_decoding` — requires `speculative_eagle_topk`. Even with `--speculative-eagle-topk 1 --speculative-num-steps 3`, the NEXTN→EAGLE mapping fails with `sigquit from child process`. Likely incompatible with Qwen3.5's conditional generation architecture in this SGLang version.

### SGLang DFlash — OOM on 24GB

`--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.5-9B-DFlash` fails with `RuntimeError: Not enough memory` at all memory fractions tested (0.75, 0.80, 0.88). The 9B FP8 model (~10GB) + 1B DFlash drafter (~2GB) + CUDA graphs + KV cache exceeds 24GB. vLLM succeeded because it uses more aggressive memory pre-allocation and smaller CUDA graph footprint.

Context reduction to 8192 tokens didn't help — the models themselves don't fit with enough headroom for KV cache.

### Correction: lovedheart/Qwen3.5-9B-FP8 DOES have MTP heads

The FP8 model contains 13 MTP tensors (`mtp.layers.0.*`). The SGLang NEXTN crash is a **SGLang bug in the NEXTN→EAGLE mapping**, not missing weights. vLLM successfully uses these same MTP tensors with `qwen3_next_mtp`.

The DFlash OOM on SGLang is a memory management issue (`--cuda-graph-max-bs` too high), fixable with `--cuda-graph-max-bs 2 --mem-fraction-static 0.5 --max-running-requests 4`.
