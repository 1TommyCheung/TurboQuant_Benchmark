# tqbench Modular Benchmark Framework

**Date:** 2026-05-23
**Status:** Design

## Problem

The current benchmark harness (`bench_embeddings_turbo/`) hard-couples inference server logic (Ollama, vLLM, llama.cpp, Gemini API, HuggingFace) with a single benchmark type (embedding quality/speed). Adding a new benchmark (e.g., text generation throughput, instruction-following quality) requires duplicating server connection boilerplate. The model config mixes server deployment details with model identity.

## Goals

1. **Shared server deployment config.** A single `servers.yaml` defines where inference servers live (type, host, port). A server may host multiple models for different purposes (embeddings, LLM, audio, video). Benchmarks reference servers by name.
2. **Fully isolated benchmarks.** Each benchmark owns its client code, model lineup, eval logic, runners, and reports. Changing one benchmark cannot break another. No shared Backend protocol, no shared client library.
3. **Reports are fully per-benchmark.** Each benchmark owns its report generation — format, templates, and output.

## Non-goals

- Shared Backend protocol or client classes across benchmarks.
- Shared `common/` utility library (benchmarks copy what they need).
- Shared orchestrator across benchmarks (each has its own `run_all`).
- Web UI or dashboard.
- Changing the existing `bench_embeddings_turbo/` directory (kept frozen as reference).

## Design principle: share config, not code

The only shared artifact is `config/servers.yaml` — a flat file describing server deployments. Each benchmark writes its own HTTP client code because the API interaction is benchmark-specific:

- An embedding client does batch `/v1/embeddings` calls with concurrent workers and L2-normalizes output.
- A generation client streams `/v1/completions` with temperature and stop tokens.
- An audio transcription client posts binary to a multipart endpoint.

These share a server address, not behavior. A shared `Backend` class with `embed()` + `generate()` + `chat()` is a false abstraction that couples unrelated code paths — a bug fix to generation retry logic should never change embedding behavior.

## Architecture

### What's shared

```
tqbench/
  config/
    servers.yaml           # the ONLY shared artifact
```

### What's per-benchmark

Everything else. Each benchmark is a fully self-contained Python package.

### Package layout

```
tqbench/
  __init__.py
  py.typed
  cli.py                      # `python -m tqbench run embeddings --model ...`

  config/
    servers.yaml               # shared server deployment config

  benchmarks/
    __init__.py                # benchmark discovery (scans sub-packages for MANIFEST)

    embeddings/
      __init__.py              # MANIFEST dict
      models.yaml              # model identity + server reference
      clients.py               # OllamaEmbedClient, VLLMEmbedClient, etc.
      models.py                # ModelSpec dataclass, registry, load/resolve
      data/                    # eval queries, chunk samples
      indexes/                 # per-model LanceDB tables (gitignored)
      runners/
        __init__.py
        run_all.py
        embed_corpus.py
        eval_quality.py
        eval_speed.py
      eval/
        __init__.py
        stack.py               # vector_only, bm25, rrf, hybrid retrieve
        io_lance.py
        snapshot.py
        pool_judge.py
        leakage.py
        source_weights.py
        metrics.py             # recall@k, MRR, NDCG, bootstrap CI
        sampling.py
        perturbations.py
        scoring.py
        schemas.py
        source_weights.py
      reports/                 # output dir (gitignored)
      templates/               # Jinja2 HTML templates
      report.py                # report builder
    # future benchmarks go here as sibling packages

pyproject.toml
```

### Existing code stays frozen

`bench_embeddings_turbo/` remains in the repo untouched. `tqbench/benchmarks/embeddings/` is ported from it. No symlinks, no imports across the boundary.

## Server Configuration (`config/servers.yaml`)

The shared config describes deployment topology only — where servers are, what type they are. A single server may host models used by multiple benchmarks (e.g., an Ollama instance serving both an embedding model and a chat model).

```yaml
servers:
  ollama-local:
    type: ollama
    host: http://127.0.0.1:11434

  turboquant-local:
    type: llamacpp
    host: http://127.0.0.1:8080

  vllm-docker:
    type: vllm
    host: http://127.0.0.1:8800

  vllm-audio:
    type: vllm
    host: http://127.0.0.1:8801

  gemini:
    type: gemini_api
    # api_key from GEMINI_API_KEY env var

  hf-local:
    type: hf
    device: cuda
```

Fields per server: `type` (string, one of: `ollama`, `vllm`, `llamacpp`, `gemini_api`, `hf`) and `host` (URL). That's it. No capabilities, no model lists, no client config. The server config answers one question: "how do I connect to this thing?"

Multiple servers of the same type are supported (e.g., two vLLM instances on different ports for different GPU allocations).

## Server Config Loader

A tiny shared utility reads `servers.yaml` and returns a dict. No classes, no protocol, no registry.

```python
# tqbench/config/__init__.py
from pathlib import Path
import yaml

SERVERS_PATH = Path(__file__).parent / "servers.yaml"

def load_servers() -> dict:
    return yaml.safe_load(SERVERS_PATH.read_text())["servers"]

def get_server(name: str) -> dict:
    servers = load_servers()
    if name not in servers:
        raise KeyError(f"Unknown server '{name}'. Known: {list(servers.keys())}")
    return servers[name]
```

Benchmarks call `get_server("ollama-local")` to get `{"type": "ollama", "host": "http://127.0.0.1:11434"}`, then use that host in their own client code.

## Benchmark Structure

### Manifest

Each benchmark sub-package exposes a `MANIFEST` dict in its `__init__.py`:

```python
MANIFEST = {
    "name": "embeddings",
    "description": "Embedding model quality and speed bake-off",
    "entry": "runners.run_all:main",
}
```

No `required_capabilities` — that concept belonged to the shared-Backend design. Each benchmark knows what it needs and validates at runtime in its own client code.

### Per-benchmark `models.yaml`

Model identity + server reference. Backend-specific fields (`ollama_model`, `hf_repo`, `llamacpp_model`) stay on the model entry since they describe which weights to load, not how to connect.

```yaml
candidates:
  - id: qwen3-embedding-8b-q8-ollama
    server: ollama-local
    ollama_model: qwen3-embedding:8b-q8_0
    dim: 4096
    max_ctx_tokens: 32768
    precision: q8_gguf

  - id: qwen3-embedding-8b-fp8-vllm
    server: vllm-docker
    hf_repo: maywell/Qwen3-Embedding-8B-FP8-Dynamic
    dim: 4096
    max_ctx_tokens: 8192
    precision: fp8_dynamic

  - id: gemini-embedding-001
    server: gemini
    dim: 3072
    max_ctx_tokens: 2048
    precision: api

baseline_dim: 3072
```

### Per-benchmark `clients.py`

Each benchmark writes its own client classes. For the embeddings benchmark, these are extracted from the current `_OllamaEmbedder`, `_VLLMEmbedder`, etc. — but they live inside the benchmark, not in a shared package.

```python
# tqbench/benchmarks/embeddings/clients.py
from tqbench.config import get_server

class OllamaEmbedClient:
    def __init__(self, model_spec):
        server = get_server(model_spec.server)
        self.host = server["host"]
        self.ollama_model = model_spec.ollama_model
        # ... embedding-specific httpx client setup

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        # ... embedding-specific: /api/embed, L2 normalize
```

A future generation benchmark writes its own:

```python
# tqbench/benchmarks/generation/clients.py
from tqbench.config import get_server

class OllamaGenerateClient:
    def __init__(self, model_spec):
        server = get_server(model_spec.server)
        self.host = server["host"]
        # ... generation-specific setup

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # ... generation-specific: /api/generate, streaming, stop tokens
```

Both read `ollama-local` from `servers.yaml` for the host. Their client code is completely independent.

### Per-benchmark `models.py`

Each benchmark defines its own `ModelSpec` dataclass with only the fields it needs:

```python
# tqbench/benchmarks/embeddings/models.py
@dataclass(frozen=True)
class ModelSpec:
    id: str
    server: str                # references servers.yaml
    dim: int
    max_ctx_tokens: int
    precision: str
    hf_repo: str | None = None
    ollama_model: str | None = None
    llamacpp_model: str | None = None
    quantization: str | None = None
    vram_estimate_gb: float | None = None
    notes: str | None = None
```

A generation benchmark's `ModelSpec` would have different fields (no `dim`, adds `vocab_size`, `supports_streaming`, etc.).

### Per-benchmark reports

Each benchmark fully owns its report pipeline:
- `report.py` — builder script, reads from its own `reports/raw/` and writes to `reports/`
- `templates/` — Jinja2 HTML templates (if applicable)
- Output format, charts, comparisons, summary JSON are all benchmark-specific

### Per-benchmark eval utilities

Metrics, sampling, scoring, and all eval logic live inside the benchmark:

| Module | Responsibility |
|---|---|
| `eval/metrics.py` | recall@k, MRR, NDCG, bootstrap CI |
| `eval/sampling.py` | Stratified sampling, token-budgeted batches |
| `eval/perturbations.py` | Typo injection, paraphrase, truncation |
| `eval/scoring.py` | Weighted scoring framework |
| `eval/schemas.py` | ChunkRecord, query dataclasses |
| `eval/stack.py` | vector_only, bm25, rrf, hybrid retrieve |
| `eval/io_lance.py` | LanceDB read/write |
| `eval/snapshot.py` | Frozen data snapshot paths |

A future benchmark copies patterns it likes but is free to diverge completely.

## CLI

```bash
# Run a specific benchmark
python -m tqbench run embeddings
python -m tqbench run embeddings --model qwen3-embedding-8b-q8-ollama

# List available benchmarks
python -m tqbench list

# Check server health (the one shared concern)
python -m tqbench servers
python -m tqbench servers ollama-local
```

The `servers` command pings each server's health endpoint (type-aware: `/api/tags` for Ollama, `/v1/models` for vLLM/llama.cpp, etc.). This is the only place where server-type-specific logic lives outside a benchmark — it's a convenience CLI, not a dependency.

## Migration Plan (embeddings benchmark)

Port from `bench_embeddings_turbo/` into `tqbench/benchmarks/embeddings/`:

| Source | Destination |
|---|---|
| `src/bench/models.py` (embedder classes) | `tqbench/benchmarks/embeddings/clients.py` |
| `src/bench/models.py` (ModelSpec, registry) | `tqbench/benchmarks/embeddings/models.py` |
| `config/models.yaml` (server fields) | `tqbench/config/servers.yaml` |
| `config/models.yaml` (model fields) | `tqbench/benchmarks/embeddings/models.yaml` |
| `src/bench/metrics.py` | `tqbench/benchmarks/embeddings/eval/metrics.py` |
| `src/bench/sampling.py` | `tqbench/benchmarks/embeddings/eval/sampling.py` |
| `src/bench/perturbations.py` | `tqbench/benchmarks/embeddings/eval/perturbations.py` |
| `src/bench/scoring.py` | `tqbench/benchmarks/embeddings/eval/scoring.py` |
| `src/bench/schemas.py` | `tqbench/benchmarks/embeddings/eval/schemas.py` |
| `src/bench/stack.py` | `tqbench/benchmarks/embeddings/eval/stack.py` |
| `src/bench/io_lance.py` | `tqbench/benchmarks/embeddings/eval/io_lance.py` |
| `src/bench/snapshot.py` | `tqbench/benchmarks/embeddings/eval/snapshot.py` |
| `src/bench/pool_judge.py` | `tqbench/benchmarks/embeddings/eval/pool_judge.py` |
| `src/bench/leakage.py` | `tqbench/benchmarks/embeddings/eval/leakage.py` |
| `src/bench/source_weights.py` | `tqbench/benchmarks/embeddings/eval/source_weights.py` |
| `runners/*.py` | `tqbench/benchmarks/embeddings/runners/*.py` |
| `templates/` | `tqbench/benchmarks/embeddings/templates/` |
| Report builders | `tqbench/benchmarks/embeddings/report.py` |

## Testing

```
tests/
  test_config.py               # servers.yaml loader
  benchmarks/
    test_embeddings_clients.py  # embedding client classes
    test_embeddings_models.py   # model registry
    test_embeddings_eval.py     # metrics, scoring, eval logic
```

Existing tests in `bench_embeddings_turbo/tests/` stay frozen. New tests cover the ported code.

## What a future benchmark looks like

Adding a "generation" benchmark:

```
tqbench/benchmarks/generation/
  __init__.py              # MANIFEST: name, description, entry
  models.yaml              # generation model lineup + server refs
  clients.py               # OllamaGenerateClient, VLLMGenerateClient, ...
  models.py                # GenerationModelSpec dataclass
  runners/
    run_all.py
    eval_throughput.py     # tok/s at various batch sizes
    eval_latency.py        # p50/p95/p99
    eval_quality.py        # perplexity, MMLU, etc.
  reports/
  templates/
  report.py
```

It reads `tqbench.config.get_server()` for server addresses. Everything else is self-contained. The embeddings benchmark is unaware it exists.

## Summary of shared surface area

| Shared | Not shared |
|---|---|
| `config/servers.yaml` (server addresses) | Client code (HTTP calls, batching, normalization) |
| `config/__init__.py` (~15 lines: YAML loader) | Model specs and registry |
| `cli.py` (benchmark discovery + `servers` health check) | Eval logic, metrics, scoring |
| `benchmarks/__init__.py` (MANIFEST scanner) | Runners and orchestration |
| | Reports, templates, output |
| | Data, indexes, snapshots |

Total shared code: ~50 lines of YAML loading + CLI discovery. Everything else is owned by the benchmark that uses it.
