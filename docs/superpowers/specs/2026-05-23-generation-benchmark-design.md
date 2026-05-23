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
| SGLang | Python/CUDA | FP8 dynamic (HF) | DFlash speculative decoding, MTP (multi-token prediction) |
| beellama.cpp | C++/CUDA (llama.cpp fork) | Q8 GGUF | DFlash + TurboQuant KV cache compression |
| TurboQuant llama.cpp | C++/CUDA (llama.cpp build) | Q8 GGUF | turbo3/turbo4 KV cache compression only |
| Lucebox-Hub | C++/CUDA (custom) | GGUF | Megakernel, DFlash, PFlash (speculative prefill) |

All expose OpenAI-compatible `/v1/chat/completions` endpoints.

## Model

**Qwen3.5-9B** — the same base model across all servers:
- vLLM / SGLang: `Qwen/Qwen3.5-9B-FP8` (FP8 dynamic quantization)
- beellama / TurboQuant / Lucebox: `Qwen3.5-9B-Q8_0.gguf` (GGUF Q8)

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

**Speed: serving benchmark**

The `eval_speed.py` runner sends generation requests at controlled concurrency:

1. Load a prompt set (150 prompts across short/medium/long input lengths)
2. For each concurrency level (1, 4, 16, 64):
   - Send prompts with `stream=true`
   - Measure per-request: TTFT (time to first token), ITL (inter-token latency), total time, output tokens
   - Aggregate: median/p95/p99 TTFT, median/p95 ITL, total throughput (output tok/s)
3. Sample VRAM via nvidia-smi during the run

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
    run_all.py             # orchestrator: quality → speed → report
    eval_quality.py        # shells out to lm_eval CLI
    eval_speed.py          # streaming TTFT/ITL/throughput benchmark

  data/
    prompts_short.jsonl    # ~50 prompts, <100 tokens input
    prompts_medium.jsonl   # ~50 prompts, 100-500 tokens input
    prompts_long.jsonl     # ~50 prompts, 500-2000 tokens input

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
  - id: qwen3.5-9b-fp8-vllm
    server: vllm-docker
    model_name: Qwen/Qwen3.5-9B-FP8
    hf_repo: Qwen/Qwen3.5-9B-FP8
    max_tokens: 4096
    notes: vLLM baseline, FP8 dynamic quantization.

  - id: qwen3.5-9b-fp8-sglang
    server: sglang-local
    model_name: Qwen/Qwen3.5-9B-FP8
    hf_repo: Qwen/Qwen3.5-9B-FP8
    max_tokens: 4096
    notes: SGLang with DFlash/MTP speculative decoding.

  - id: qwen3.5-9b-q8-beellama
    server: beellama-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    notes: beellama.cpp fork with DFlash + TurboQuant KV cache.

  - id: qwen3.5-9b-q8-turboquant
    server: turboquant-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    notes: TurboQuant llama.cpp, turbo cache only.

  - id: qwen3.5-9b-q8-lucebox
    server: lucebox-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    notes: Lucebox-Hub, megakernel/DFlash/PFlash.
```

## Speed Benchmark Detail

### Prompt set

150 prompts in 3 JSONL files, each with:
```json
{"id": "short_001", "messages": [{"role": "user", "content": "..."}], "max_tokens": 256}
```

Input length distribution:
- `prompts_short.jsonl` — 50 prompts, <100 input tokens (quick Q&A, single-turn)
- `prompts_medium.jsonl` — 50 prompts, 100-500 input tokens (code gen, reasoning)
- `prompts_long.jsonl` — 50 prompts, 500-2000 input tokens (summarization, analysis)

Prompts sourced from open datasets (ShareGPT, HumanEval, GSM8K) to represent real workloads.

### Concurrency levels

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

### VRAM sampling

Background thread polls `nvidia-smi --query-gpu=memory.used` every 1 second during the speed run. Records peak VRAM per server.

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
1. **Executive summary** — quality pass/fail per server + speed winner at each concurrency level
2. **Quality table** — tinyMMLU, tinyHellaSwag, tinyARC, tinyWinogrande scores per server, delta vs vLLM baseline
3. **Speed: TTFT** — bar chart per concurrency level, all servers side by side
4. **Speed: Throughput** — line chart, throughput (tok/s) vs concurrency
5. **Speed: ITL distribution** — box plot per server
6. **Speed: Request latency** — p50/p95/p99 table
7. **VRAM usage** — bar chart
8. **Methodology** — prompt set description, hardware, server versions, config flags

## Testing

```
tests/benchmarks/
  test_generation_models.py    # ModelSpec, registry
  test_generation_clients.py   # OpenAIGenerateClient with mock HTTP
  test_generation_speed.py     # Metric aggregation functions
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
