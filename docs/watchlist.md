# Watchlist — Upstream items to monitor

This file tracks external projects, PRs, issues, and model releases whose
resolution would change our deployment options. Re-check periodically (suggest
monthly via `/schedule monthly check the TurboQuant_Benchmark watchlist for PR merges and issue updates`).

**Last checked:** 2026-05-24

| # | Item | What we want | Current state | URL | Last checked |
|---|---|---|---|---|---|
| 1 | vLLM PR #40914 | TurboQuant + MTP/EAGLE speculative decoding combo merged into upstream | OPEN. Validated by Genesis fork on Qwen3.6-35B + RTX A5000 (+32% TPS, 18/18 tool-call tests pass). No merge date. | https://github.com/vllm-project/vllm/pull/40914 | 2026-05-24 |
| 2 | vLLM PR #39995 | DFlash + FlashInfer + FP8 KV cache support | OPEN. Adds non-causal prefill to FlashInfer for DFlash draft attention with FP8 KV. Does NOT address TurboQuant — only FP8. | https://github.com/vllm-project/vllm/pull/39995 | 2026-05-24 |
| 3 | vLLM Issue #41559 | DFlash + TurboQuant structural fix (TQ hardcodes causal=True, DFlash needs causal=False) | OPEN. Suggested fix in issue thread, no PR yet. **Highest priority** — unlocks combo on vLLM. | https://github.com/vllm-project/vllm/issues/41559 | 2026-05-24 |
| 4 | vLLM Issue #40831 | TurboQuant + MTP produces degenerate output (token-cascade loops, all 3 TQ presets affected) | OPEN. Same fix as PR #40914 addresses this. | https://github.com/vllm-project/vllm/issues/40831 | 2026-05-24 |
| 5 | vLLM Issue #42808 | TurboQuant + MTP workspace allocation failure on v0.21.0 | OPEN. `AssertionError: Workspace is locked but allocation from turboquant_attn.py:879 requires 0.76 MB`. Separate from #40914. | https://github.com/vllm-project/vllm/issues/42808 | 2026-05-24 |
| 6 | Lucebox PFlash recipe for 9B target | Published `keep_ratio` + `alpha` calibrated for Qwen3.5-9B (currently only 27B is validated) | Not present in repo. Would extend PFlash VRAM win down to 9B. | https://github.com/Luce-Org/lucebox-hub | 2026-05-24 |
| 7 | Qwen3.5-9B official MTP weights | Native MTP heads exposed in the HF release (currently only community DFlash drafts) | Not released. `lovedheart/Qwen3.5-9B-FP8` preserves MTP via `text_config.mtp_num_hidden_layers: 1` but no official artifact. | https://huggingface.co/Qwen | 2026-05-24 |
| 8 | Smaller DFlash draft for 9B | A ~0.5B draft for Qwen3.5-9B target (current draft is 1.5B → 1.1 GB Q8). Halving draft size cuts combo's 1.5 GB overhead in half. | None exists. Would need community quant or distillation. | — | 2026-05-24 |
| 9 | Lucebox WSL2 performance fix | Direct binary PFlash on RTX 4090 WSL2 measured ~157s at 128K vs ~25s claim on RTX 3090 native. Root cause not isolated. | Open question. May be WSL2 GPU memory bandwidth, P-state, or BSA kernel sm_89 tuning. Could file an issue. | https://github.com/Luce-Org/lucebox-hub/issues | 2026-05-24 |
| 10 | Sandermage/genesis-vllm-patches Qwen3.5-9B support | Currently targets Qwen3.6 27B/35B. Extending to Qwen3.5-9B would give us a TQ+MTP path on vLLM without waiting for PR #40914. | Not supported. Their P67/P67b patches are arch-specific. | https://github.com/Sandermage/genesis-vllm-patches | 2026-05-24 |
| 11 | vLLM TurboQuant workspace bug at long context (no spec) | Workspace is pre-sized during warmup and locked, then 60K-token prefill needs more memory and asserts at `turboquant_attn.py:747`. Same class as #42808 but fires WITHOUT MTP. Hit this on Qwen3.5-9B-FP8 + `turboquant_k3v4_nc` + max_model_len=65536. | Not yet filed as a separate issue. May be covered by future fix to #42808 or PR #40914 workspace pre-sizing path. **Worth filing if not.** | https://github.com/vllm-project/vllm/issues/42808 | 2026-05-24 |

## What unblocks what

- **#1 merges** → can run TQ + MTP combo on vLLM (still not DFlash though)
- **#3 resolves** → can run TQ + DFlash combo on vLLM (matches beellama)
- **#6 published** → Lucebox stack (PFlash+TQ+DFlash) usable on 9B → extends long-context VRAM win to smaller model
- **#8 exists** → DFlash combo becomes attractive for 9B short outputs (currently breaks even at 4-8K context)
- **#9 isolated** → could 6× speed up our 27B + 128K runs on this 4090

## Re-check command

When the schedule fires (or you run a manual check), spawn a sonnet research subagent
with this prompt:

```
For each item in TurboQuant_Benchmark/docs/watchlist.md, check the current state
of the referenced URL. Report what's changed since 2026-05-24. Specifically:
- For PRs: merged yet? new validation? new comments?
- For issues: new PRs linking to it? closed? root cause identified?
- For model releases: new artifacts on HuggingFace?
For each item, output one line: "[changed/unchanged] item N: <summary>".
End with a recommendation: which items now warrant a follow-up benchmark run.
```

Update the "Last checked" column for items you re-verified.
