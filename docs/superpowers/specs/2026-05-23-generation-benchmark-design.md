# Generation Benchmark — Inference Server Comparison

**Date:** 2026-05-23
**Status:** Design

## Problem

We need to compare inference server architectures (vLLM, SGLang, beellama.cpp, TurboQuant llama.cpp, Lucebox-Hub) for text generation on the same model (Qwen3.5-9B). Each server has different optimization strategies (continuous batching, speculative decoding, KV cache compression, custom CUDA kernels) but all expose OpenAI-compatible APIs. We need both quality verification (quantization/backend doesn't degrade output) and speed measurement (TTFT, throughput, latency under load).

## Goals

1. Verify output quality is equivalent across all 5 servers using tinyBenchmarks (~400 examples, 15 min per server).
2. Measure serving speed (TTFT, ITL, throughput) at concurrency levels 1, 4, 16, 64.
3. Produce a comparison report (HTML) with quality table + speed charts.
4. Fit into the existing tqbench framework — shared `servers.yaml`, fully isolated benchmark package.

## Non-goals

- Full MMLU or large-scale quality eval (tinyBenchmarks is sufficient for backend parity checks).
- Training, fine-tuning, or LoRA comparison.
- Multi-GPU or distributed inference.
- Embedding evaluation (separate benchmark).

## Servers Under Test

| Server | Type | Model Format | Key Features |
|---|---|---|---|
| vLLM | Python/CUDA | FP8 dynamic (HF) | Continuous batching, PagedAttention, production baseline |
| SGLang | Python/CUDA | FP8 dynamic (HF) | DFlash, MTP (multi-token prediction), continuous batching |
| beellama.cpp | C++/CUDA (llama.cpp fork) | Q8 GGUF | DFlash + TurboQuant KV cache compression |
| TurboQuant llama.cpp | C++/CUDA (llama.cpp build) | Q8 GGUF | turbo3/turbo4 KV cache compression only |
| Lucebox-Hub | C++/CUDA (custom) | GGUF | Megakernel, DFlash, PFlash (speculative prefill) |

All expose OpenAI-compatible `/v1/chat/completions` endpoints.

## Speculative Decoding & Prefill

Three acceleration techniques are under test. Each is available on different servers:

### DFlash (speculative decoding — decode stage)

Block-diffusion drafter generates an entire block of tokens in one forward pass. Requires a trained DFlash drafter per target model. Claims ~6x lossless acceleration, ~2.5x faster than EAGLE-3.

- **Drafter:** `z-lab/Qwen3.5-9B-DFlash` (HuggingFace)
- **Available in:** vLLM (v0.20.1+ via `vllm-speculators`), SGLang (`--speculative-algorithm DFLASH`), beellama.cpp, Lucebox-Hub
- **Paper:** arXiv 2602.06036

### MTP — Multi-Token Prediction (decode stage)

Native MTP heads trained into the model predict multiple tokens per forward pass. No separate drafter needed — Qwen3.5 has MTP heads built in.

- **Available in:** vLLM (v0.17+ via `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`), SGLang
- **Not available in:** llama.cpp variants, Lucebox-Hub
- **Speedup:** ~60% throughput boost

### PFlash (speculative prefill — prefill stage)

Uses a small drafter (Qwen3-0.6B BF16) to score token importance across a long prompt, selects a compressed subset (~5% of tokens), runs only that subset through the full model's prefill. Slashes TTFT for long contexts (128K+). Training-free — no fine-tuning needed for the drafter.

- **Drafter:** Qwen3-0.6B BF16 (importance scorer)
- **Available in:** Lucebox-Hub only (vLLM has open feature request #39060)
- **Speedup:** ~10x prefill speedup over llama.cpp at 128K context
- **Paper:** Based on arXiv 2502.02789 (Cross-Family Speculative Prefill, ICML 2025)
- **Combines with DFlash:** PFlash handles prefill, DFlash handles decode — full pipeline on one GPU

## Configuration Matrix

Each row is a separate benchmark config. This lets us isolate server overhead vs. speculative decoding gains.

| Config ID | Server | Prefill | Decode | What it tests |
|---|---|---|---|---|
| `qwen3.5-9b-fp8-vllm` | vLLM | standard | standard | Baseline — pure server overhead |
| `qwen3.5-9b-fp8-vllm-mtp` | vLLM | standard | MTP (native) | Native MTP value on vLLM |
| `qwen3.5-9b-fp8-vllm-dflash` | vLLM | standard | DFlash | DFlash value on vLLM |
| `qwen3.5-9b-fp8-sglang` | SGLang | standard | standard | SGLang baseline |
| `qwen3.5-9b-fp8-sglang-mtp` | SGLang | standard | MTP (native) | Native MTP value on SGLang |
| `qwen3.5-9b-fp8-sglang-dflash` | SGLang | standard | DFlash | DFlash value on SGLang |
| `qwen3.5-9b-q8-beellama-dflash` | beellama.cpp | standard | DFlash + TQ KV | llama.cpp fork with DFlash |
| `qwen3.5-9b-q8-turboquant` | TurboQuant | standard | standard (TQ KV) | TurboQuant KV cache only |
| `qwen3.5-9b-q8-lucebox` | Lucebox | standard | DFlash | Lucebox without PFlash |
| `qwen3.5-9b-q8-lucebox-pflash` | Lucebox | **PFlash** | DFlash | Full Lucebox pipeline (PFlash + DFlash) |

Quality eval (tinyBenchmarks) only needs to run once per unique model format — FP8 configs share one quality run, Q8 configs share another. Speed eval runs on every config.

## Model

**Qwen3.5-9B** — the same base model across all servers:
- vLLM / SGLang: `Qwen/Qwen3.5-9B-FP8` (FP8 dynamic quantization)
- beellama / TurboQuant / Lucebox: `Qwen3.5-9B-Q8_0.gguf` (GGUF Q8)

**DFlash drafter:** `z-lab/Qwen3.5-9B-DFlash` (used by DFlash configs)
**PFlash drafter:** `Qwen3-0.6B` BF16 (used by Lucebox PFlash config only)

Both FP8 and Q8 are 8-bit quantizations. Quality differences between them (if any) will show up in the tinyBenchmarks results, which is exactly the point — we want to know if a backend degrades quality.

## Architecture

### Single client class

All 5 servers speak the same OpenAI chat completions API. One client class handles all of them:

```python
class OpenAIGenerateClient:
    def __init__(self, spec: ModelSpec, server: dict):
        self.base_url = server["host"]
        self.model = spec.model_name
        # httpx client with streaming support

    def generate(self, messages: list[dict], max_tokens: int = 256,
                 temperature: float = 0.0) -> GenerateResult:
        # POST /v1/chat/completions, non-streaming
        # Returns: text, token counts, total_time

    def generate_stream(self, messages: list[dict], max_tokens: int = 256,
                        temperature: float = 0.0) -> StreamResult:
        # POST /v1/chat/completions with stream=true
        # Returns: text, token counts, ttft, itl_list, total_time

    def health(self) -> bool:
        # GET /v1/models
```

### Two measurement axes

**Quality: tinyBenchmarks via lm-evaluation-harness**

The `eval_quality.py` runner shells out to `lm_eval` CLI:

```bash
lm_eval --model local-chat-completions \
  --model_args model=<model_name>,base_url=<host>/v1,tokenizer_backend=huggingface \
  --tasks tinyMMLU,tinyHellaSwag,tinyARC,tinyWinogrande \
  --output_path reports/raw/<date>_<id>_quality.json
```

This avoids reimplementing eval logic — lm-evaluation-harness handles tokenization, scoring, and normalization. We just parse the output JSON.

Tasks (~400 total examples):
- `tinyMMLU` — knowledge across 57 subjects (~100 examples)
- `tinyHellaSwag` — commonsense reasoning (~100 examples)
- `tinyARC` — science reasoning (~100 examples)
- `tinyWinogrande` — coreference resolution (~100 examples)

Each is proven to predict full benchmark scores within 2%.

**Speed: serving benchmark (two sub-tests)**

The speed benchmark has two parts:

**A. Throughput & latency under load (`eval_speed.py`)**

Sends generation requests at controlled concurrency using 150 prompts (short/medium/long):

1. Load prompt set (150 prompts across short/medium/long input lengths)
2. For each concurrency level (1, 4, 16, 64):
   - Send prompts with `stream=true`
   - Measure per-request: TTFT, ITL, total time, output tokens
   - Aggregate: median/p95/p99 TTFT, median/p95 ITL, total throughput (output tok/s)
3. Sample VRAM via nvidia-smi during the run

**B. Long-context TTFT (`eval_ttft_longctx.py`)**

Isolates prefill performance at increasing input lengths. This is where PFlash's advantage becomes visible.

1. For each context length (1K, 4K, 8K, 32K, 128K tokens):
   - Send 10 identical-length prompts (padded/truncated from a long document), `max_tokens=1`
   - Measure TTFT only (we request 1 output token — decode time is negligible)
   - Aggregate: median, p95 TTFT
2. Concurrency = 1 (isolate pure prefill, no batching effects)

This produces a TTFT-vs-context-length curve per config. Standard servers should show linear TTFT growth; PFlash should show sublinear (only processes ~5% of tokens).

Prompts are fixed across all servers for reproducibility. Temperature 0.0 for deterministic output.

### Package layout

```
tqbench/benchmarks/generation/
  __init__.py              # MANIFEST
  models.yaml              # Qwen3.5-9B variants per server
  models.py                # ModelSpec dataclass + registry
  clients.py               # OpenAIGenerateClient

  runners/
    __init__.py
    run_all.py             # orchestrator: quality → speed → ttft → report
    eval_quality.py        # shells out to lm_eval CLI
    eval_speed.py          # streaming TTFT/ITL/throughput benchmark
    eval_ttft_longctx.py   # long-context TTFT isolation test

  data/
    prompts_short.jsonl    # ~50 prompts, <100 tokens input
    prompts_medium.jsonl   # ~50 prompts, 100-500 tokens input
    prompts_long.jsonl     # ~50 prompts, 500-2000 tokens input
    prompts_longctx.jsonl  # 10 prompts at each of 1K/4K/8K/32K/128K tokens

  reports/                 # output dir (gitignored)
    raw/                   # per-server JSON results
  templates/
    report.html.j2         # Jinja2 HTML template
  report.py                # HTML comparison report builder
```

### Data classes

```python
@dataclass(frozen=True)
class ModelSpec:
    id: str
    server: str            # references servers.yaml
    model_name: str        # model name passed to the API
    max_tokens: int = 4096
    hf_repo: str | None = None
    gguf_file: str | None = None
    spec_decode: str | None = None    # "dflash", "mtp", or None
    spec_prefill: str | None = None   # "pflash" or None
    drafter_repo: str | None = None   # HF repo or GGUF for DFlash drafter
    quality_group: str | None = None  # configs sharing a quality run (e.g. "fp8", "q8")
    notes: str | None = None

@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_time_s: float

@dataclass
class StreamResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float          # time to first token
    itl_ms: list[float]    # inter-token latencies in milliseconds
    total_time_s: float
```

The `spec_decode`, `spec_prefill`, and `drafter_repo` fields are metadata — the benchmark doesn't configure the server (the user starts each server with the right flags). These fields are used for labeling in reports and for deciding which configs share quality runs (`quality_group`).

## Server Configuration

Add to `tqbench/config/servers.yaml`:

```yaml
  sglang-local:
    type: sglang
    host: http://127.0.0.1:30000

  beellama-local:
    type: llamacpp
    host: http://127.0.0.1:8081

  lucebox-local:
    type: lucebox
    host: http://127.0.0.1:8082
```

Existing entries (`vllm-docker`, `turboquant-local`) are reused.

## Per-benchmark `models.yaml`

```yaml
candidates:
  # ── vLLM configs ──
  - id: qwen3.5-9b-fp8-vllm
    server: vllm-docker
    model_name: Qwen/Qwen3.5-9B-FP8
    hf_repo: Qwen/Qwen3.5-9B-FP8
    max_tokens: 4096
    quality_group: fp8
    notes: vLLM baseline, no speculative decoding.

  - id: qwen3.5-9b-fp8-vllm-mtp
    server: vllm-docker
    model_name: Qwen/Qwen3.5-9B-FP8
    hf_repo: Qwen/Qwen3.5-9B-FP8
    max_tokens: 4096
    spec_decode: mtp
    quality_group: fp8
    notes: "vLLM + native MTP. Launch: --speculative-config '{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}'"

  - id: qwen3.5-9b-fp8-vllm-dflash
    server: vllm-docker
    model_name: Qwen/Qwen3.5-9B-FP8
    hf_repo: Qwen/Qwen3.5-9B-FP8
    max_tokens: 4096
    spec_decode: dflash
    drafter_repo: z-lab/Qwen3.5-9B-DFlash
    quality_group: fp8
    notes: vLLM + DFlash via vllm-speculators.

  # ── SGLang configs ──
  - id: qwen3.5-9b-fp8-sglang
    server: sglang-local
    model_name: Qwen/Qwen3.5-9B-FP8
    hf_repo: Qwen/Qwen3.5-9B-FP8
    max_tokens: 4096
    quality_group: fp8
    notes: SGLang baseline, no speculative decoding.

  - id: qwen3.5-9b-fp8-sglang-mtp
    server: sglang-local
    model_name: Qwen/Qwen3.5-9B-FP8
    hf_repo: Qwen/Qwen3.5-9B-FP8
    max_tokens: 4096
    spec_decode: mtp
    quality_group: fp8
    notes: SGLang + native MTP.

  - id: qwen3.5-9b-fp8-sglang-dflash
    server: sglang-local
    model_name: Qwen/Qwen3.5-9B-FP8
    hf_repo: Qwen/Qwen3.5-9B-FP8
    max_tokens: 4096
    spec_decode: dflash
    drafter_repo: z-lab/Qwen3.5-9B-DFlash
    quality_group: fp8
    notes: "SGLang + DFlash. Launch: --speculative-algorithm DFLASH --speculative-model z-lab/Qwen3.5-9B-DFlash"

  # ── llama.cpp configs ──
  - id: qwen3.5-9b-q8-beellama-dflash
    server: beellama-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    spec_decode: dflash
    quality_group: q8
    notes: "beellama.cpp with DFlash + TurboQuant KV. Launch: --spec-type dflash --spec-draft-model <drafter.gguf>"

  - id: qwen3.5-9b-q8-turboquant
    server: turboquant-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    quality_group: q8
    notes: TurboQuant llama.cpp, turbo cache only, no speculative decoding.

  # ── Lucebox configs ──
  - id: qwen3.5-9b-q8-lucebox
    server: lucebox-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    spec_decode: dflash
    quality_group: q8
    notes: Lucebox-Hub with DFlash, no PFlash.

  - id: qwen3.5-9b-q8-lucebox-pflash
    server: lucebox-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    spec_decode: dflash
    spec_prefill: pflash
    quality_group: q8
    notes: Lucebox-Hub full pipeline — PFlash (speculative prefill) + DFlash (speculative decode).
```

Quality eval runs once per `quality_group` (fp8 and q8). Speed eval runs on every config. The `notes` field includes server launch flags for reference.

## Speed Benchmark Detail

### Prompt set (throughput test)

150 prompts in 3 JSONL files, each with:
```json
{"id": "short_001", "messages": [{"role": "user", "content": "..."}], "max_tokens": 256}
```

Input length distribution:
- `prompts_short.jsonl` — 50 prompts, <100 input tokens (quick Q&A, single-turn)
- `prompts_medium.jsonl` — 50 prompts, 100-500 input tokens (code gen, reasoning)
- `prompts_long.jsonl` — 50 prompts, 500-2000 input tokens (summarization, analysis)

Prompts sourced from open datasets (ShareGPT, HumanEval, GSM8K) to represent real workloads.

### Prompt set (long-context TTFT test)

```json
{"id": "longctx_1k_01", "messages": [{"role": "user", "content": "..."}], "max_tokens": 1, "target_input_tokens": 1024}
```

- `prompts_longctx.jsonl` — 10 prompts at each target length: 1K, 4K, 8K, 32K, 128K tokens
- Prompts are long documents (book excerpts, code files, legal text) truncated/padded to target length
- `max_tokens=1` — we only care about prefill time, not generation

### Concurrency levels (throughput test)

For each concurrency level (1, 4, 16, 64):
1. Queue all 150 prompts
2. Send N concurrent requests using asyncio + httpx async client
3. Each request uses `stream=true` to capture TTFT and per-token timing
4. Record per-request metrics
5. Aggregate:
   - Throughput: total output tokens / total wall time (tok/s)
   - TTFT: median, p95, p99
   - ITL: median, p95
   - Request latency: median, p95

### Long-context TTFT test

For each context length (1K, 4K, 8K, 32K, 128K):
1. Send 10 prompts sequentially (concurrency=1)
2. Each request: `stream=true`, `max_tokens=1`
3. Measure TTFT only
4. Aggregate: median, p95

This produces a TTFT-vs-context-length curve. Standard servers show linear growth; PFlash should show sublinear (~5% of tokens processed).

### VRAM sampling

Background thread polls `nvidia-smi --query-gpu=memory.used` every 1 second during both speed tests. Records peak VRAM per config.

## Quality Benchmark Detail

### lm-evaluation-harness integration

```bash
pip install lm-eval[api]
```

The runner constructs and executes the CLI command:
```bash
lm_eval --model local-chat-completions \
  --model_args model={model_name},base_url={host}/v1,tokenizer_backend=huggingface \
  --tasks tinyMMLU,tinyHellaSwag,tinyARC,tinyWinogrande \
  --batch_size 1 \
  --output_path {reports_dir}/raw/{date}_{model_id}_quality.json
```

The runner then parses the output JSON to extract per-task scores and writes a normalized summary.

### Quality pass/fail

A server "passes" quality if its tinyBenchmark scores are within 2 percentage points of the vLLM baseline on all 4 tasks. This is the expected accuracy of tinyBenchmarks itself — differences larger than 2pp indicate real quality degradation from the backend.

## Report

Fully owned by this benchmark. HTML + Plotly, generated by `report.py`.

Sections:
1. **Executive summary** — quality pass/fail per quality group + speed winner at each concurrency level + long-context TTFT winner
2. **Configuration matrix** — table of all 10 configs with server, prefill method, decode method
3. **Quality table** — tinyMMLU, tinyHellaSwag, tinyARC, tinyWinogrande scores per quality group (fp8, q8), delta vs vLLM baseline
4. **Speed: TTFT** — bar chart per concurrency level, all configs side by side
5. **Speed: Throughput** — line chart, throughput (tok/s) vs concurrency
6. **Speed: ITL distribution** — box plot per config
7. **Speed: Request latency** — p50/p95/p99 table
8. **Long-context TTFT** — line chart, TTFT (ms) vs context length (1K→128K) per config. This is where PFlash's sublinear curve should be visible.
9. **VRAM usage** — bar chart per config
10. **Speculative decoding comparison** — grouped bar chart: baseline vs MTP vs DFlash for each server, isolating the spec-decode gain
11. **Methodology** — prompt set description, hardware, server versions, config flags, drafter models

## Testing

```
tests/benchmarks/
  test_generation_models.py    # ModelSpec, registry, quality_group logic
  test_generation_clients.py   # OpenAIGenerateClient with mock HTTP (generate + stream)
  test_generation_speed.py     # Metric aggregation functions (TTFT, ITL, throughput)
```

No live-server tests — all client tests use mocked HTTP responses.

## Dependencies

Added to `pyproject.toml` as optional:
```toml
generation = [
    "httpx>=0.27",
    "numpy>=1.24",
    "plotly>=5.20",
    "jinja2>=3.1",
]
generation-eval = [
    "lm-eval[api]>=0.4",
]
```
