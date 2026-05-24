# Generation Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `tqbench/benchmarks/generation/` package that compares 5 inference servers (vLLM, SGLang, beellama.cpp, TurboQuant, Lucebox-Hub) across 10 configurations on Qwen3.5-9B, measuring quality (tinyBenchmarks) and speed (TTFT, ITL, throughput, long-context TTFT).

**Architecture:** Fully isolated benchmark within the existing tqbench framework. One `OpenAIGenerateClient` class talks to all servers via `/v1/chat/completions`. Quality eval shells out to `lm_eval` CLI. Speed eval uses async httpx with streaming to measure TTFT/ITL/throughput. Long-context TTFT test isolates prefill performance at 1K→128K tokens.

**Tech Stack:** Python 3.11+, httpx (sync + async), PyYAML, numpy, asyncio, Plotly, Jinja2, lm-evaluation-harness

**Spec:** `docs/superpowers/specs/2026-05-23-generation-benchmark-design.md`

---

## File Map

### New files in tqbench/benchmarks/generation/

| File | Responsibility |
|---|---|
| `__init__.py` | MANIFEST dict |
| `models.yaml` | 10 configs across 5 servers with spec_decode/spec_prefill metadata |
| `models.py` | ModelSpec dataclass, registry, quality_group helpers |
| `clients.py` | OpenAIGenerateClient (generate, generate_stream, health) |
| `speed_metrics.py` | Aggregation functions for TTFT, ITL, throughput stats |
| `vram.py` | nvidia-smi VRAM sampling in background thread |
| `runners/__init__.py` | Package marker |
| `runners/run_all.py` | Orchestrator |
| `runners/eval_quality.py` | Drives lm_eval CLI, parses output |
| `runners/eval_speed.py` | Async throughput/latency benchmark |
| `runners/eval_ttft_longctx.py` | Long-context TTFT isolation test |
| `data/` | Prompt JSONL files (gitignored except seeds) |
| `reports/` | Output dir (gitignored) |
| `templates/report.html.j2` | Jinja2 HTML report template |
| `report.py` | HTML report builder |
| `.gitignore` | Ignore generated data/reports |

### Modified files

| File | Change |
|---|---|
| `tqbench/config/servers.yaml` | Add sglang-local, beellama-local, lucebox-local |
| `tqbench/pyproject.toml` | Add generation and generation-eval optional deps |

### Test files

| File | Tests for |
|---|---|
| `tests/benchmarks/test_generation_models.py` | ModelSpec, registry, quality_group |
| `tests/benchmarks/test_generation_clients.py` | OpenAIGenerateClient with mocked HTTP |
| `tests/benchmarks/test_generation_speed.py` | Speed metric aggregation functions |

---

## Task 1: Add generation servers + deps to shared config

**Files:**
- Modify: `tqbench/config/servers.yaml`
- Modify: `tqbench/pyproject.toml`

- [ ] **Step 1: Add new servers to servers.yaml**

Append to `tqbench/config/servers.yaml`:

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

- [ ] **Step 2: Add generation deps to pyproject.toml**

Add these optional dependency groups after the existing `embeddings-judge` line in `tqbench/pyproject.toml`:

```toml
generation = [
    "numpy>=1.24",
    "plotly>=5.20",
    "jinja2>=3.1",
]
generation-eval = [
    "lm-eval>=0.4",
]
```

- [ ] **Step 3: Run existing config tests**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/test_config.py -v`
Expected: All pass (existing + new servers)

- [ ] **Step 4: Commit**

```bash
git add tqbench/config/servers.yaml tqbench/pyproject.toml
git commit -m "feat: add sglang, beellama, lucebox servers + generation deps"
```

---

## Task 2: Generation MANIFEST + models.yaml + ModelSpec + registry

**Files:**
- Create: `tqbench/benchmarks/generation/__init__.py`
- Create: `tqbench/benchmarks/generation/models.yaml`
- Create: `tqbench/benchmarks/generation/models.py`
- Create: `tests/benchmarks/test_generation_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_generation_models.py
from __future__ import annotations
import pytest
from tqbench.benchmarks.generation.models import (
    load_registry, get_candidate, ModelSpec, quality_groups,
)


def test_registry_loads_10_candidates():
    reg = load_registry()
    assert len(reg) == 10
    assert all(isinstance(m, ModelSpec) for m in reg)


def test_every_candidate_has_server_and_model_name():
    for m in load_registry():
        assert m.server, f"'{m.id}' missing server"
        assert m.model_name, f"'{m.id}' missing model_name"


def test_get_candidate_by_id():
    spec = get_candidate("qwen3.5-9b-fp8-vllm")
    assert spec.server == "vllm-docker"
    assert spec.model_name == "Qwen/Qwen3.5-9B-FP8"
    assert spec.spec_decode is None
    assert spec.quality_group == "fp8"


def test_get_candidate_with_spec_decode():
    spec = get_candidate("qwen3.5-9b-fp8-vllm-dflash")
    assert spec.spec_decode == "dflash"
    assert spec.drafter_repo == "z-lab/Qwen3.5-9B-DFlash"


def test_get_candidate_with_pflash():
    spec = get_candidate("qwen3.5-9b-q8-lucebox-pflash")
    assert spec.spec_decode == "dflash"
    assert spec.spec_prefill == "pflash"


def test_get_candidate_missing_raises():
    with pytest.raises(KeyError):
        get_candidate("does-not-exist")


def test_quality_groups():
    groups = quality_groups()
    assert "fp8" in groups
    assert "q8" in groups
    assert len(groups["fp8"]) >= 1
    assert len(groups["q8"]) >= 1
    # All configs in a group share the same quality_group
    for gname, specs in groups.items():
        for s in specs:
            assert s.quality_group == gname
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_generation_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create tqbench/benchmarks/generation/__init__.py**

```python
"""Generation benchmark — inference server comparison."""

MANIFEST = {
    "name": "generation",
    "description": "Inference server generation speed and quality comparison",
    "entry": "tqbench.benchmarks.generation.runners.run_all:main",
}
```

- [ ] **Step 4: Create tqbench/benchmarks/generation/models.yaml**

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
    notes: vLLM + native MTP.

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
    notes: SGLang + DFlash.

  # ── llama.cpp configs ──
  - id: qwen3.5-9b-q8-beellama-dflash
    server: beellama-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    spec_decode: dflash
    quality_group: q8
    notes: beellama.cpp with DFlash + TurboQuant KV.

  - id: qwen3.5-9b-q8-turboquant
    server: turboquant-local
    model_name: qwen3.5-9b
    gguf_file: Qwen3.5-9B-Q8_0.gguf
    max_tokens: 4096
    quality_group: q8
    notes: TurboQuant llama.cpp, turbo cache only.

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
    notes: Lucebox-Hub full pipeline — PFlash + DFlash.
```

- [ ] **Step 5: Create tqbench/benchmarks/generation/models.py**

```python
"""Model registry for the generation benchmark."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    server: str
    model_name: str
    max_tokens: int = 4096
    hf_repo: str | None = None
    gguf_file: str | None = None
    spec_decode: str | None = None
    spec_prefill: str | None = None
    drafter_repo: str | None = None
    quality_group: str | None = None
    notes: str | None = None


def load_registry() -> list[ModelSpec]:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    return [ModelSpec(**c) for c in raw["candidates"]]


def get_candidate(model_id: str) -> ModelSpec:
    for c in load_registry():
        if c.id == model_id:
            return c
    raise KeyError(model_id)


def quality_groups() -> dict[str, list[ModelSpec]]:
    """Group configs by quality_group — quality eval runs once per group."""
    groups: dict[str, list[ModelSpec]] = defaultdict(list)
    for c in load_registry():
        if c.quality_group:
            groups[c.quality_group].append(c)
    return dict(groups)


def load_client(model_id: str):
    """Return a client with .generate() and .generate_stream() methods."""
    from tqbench.benchmarks.generation.clients import build_client
    spec = get_candidate(model_id)
    return build_client(spec)
```

- [ ] **Step 6: Run tests**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_generation_models.py -v`
Expected: 7 passed

- [ ] **Step 7: Verify benchmark discovery picks up generation**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m tqbench list`
Expected: Shows both `embeddings` and `generation`

- [ ] **Step 8: Commit**

```bash
git add tqbench/benchmarks/generation/__init__.py tqbench/benchmarks/generation/models.yaml tqbench/benchmarks/generation/models.py tests/benchmarks/test_generation_models.py
git commit -m "feat: generation benchmark MANIFEST, ModelSpec, 10-config registry"
```

---

## Task 3: OpenAIGenerateClient

**Files:**
- Create: `tqbench/benchmarks/generation/clients.py`
- Create: `tests/benchmarks/test_generation_clients.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_generation_clients.py
from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, patch
from tqbench.benchmarks.generation.models import ModelSpec
from tqbench.benchmarks.generation.clients import (
    OpenAIGenerateClient, GenerateResult, StreamResult, build_client,
)


def _spec(**kw) -> ModelSpec:
    defaults = dict(id="test", server="vllm-docker", model_name="test-model",
                    max_tokens=256)
    defaults.update(kw)
    return ModelSpec(**defaults)


def test_generate_returns_result():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello world"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_client.post.return_value = mock_resp
    mock_client.get.return_value = MagicMock(status_code=200)

    client = OpenAIGenerateClient.__new__(OpenAIGenerateClient)
    client.model = "test-model"
    client.client = mock_client

    result = client.generate([{"role": "user", "content": "hi"}])
    assert isinstance(result, GenerateResult)
    assert result.text == "Hello world"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_time_s >= 0


def test_generate_stream_returns_stream_result():
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}\n\n',
        b'data: [DONE]\n\n',
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(
        line.decode().strip() for chunk in chunks for line in chunk.strip().split(b"\n") if line.strip()
    )

    mock_client = MagicMock()
    mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_resp)
    mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

    client = OpenAIGenerateClient.__new__(OpenAIGenerateClient)
    client.model = "test-model"
    client.client = mock_client

    result = client.generate_stream([{"role": "user", "content": "hi"}])
    assert isinstance(result, StreamResult)
    assert result.text == "Hello world"
    assert result.ttft_s >= 0
    assert len(result.itl_ms) >= 1


def test_health_returns_bool():
    mock_client = MagicMock()
    mock_client.get.return_value = MagicMock(status_code=200)

    client = OpenAIGenerateClient.__new__(OpenAIGenerateClient)
    client.client = mock_client

    assert client.health() is True


def test_build_client_resolves_server():
    spec = _spec()
    # build_client calls get_server which reads servers.yaml
    # vllm-docker exists in servers.yaml, but vLLM isn't running
    # so it should raise on the health check
    with pytest.raises(Exception):
        build_client(spec)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_generation_clients.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create tqbench/benchmarks/generation/clients.py**

```python
"""OpenAI-compatible generation client — one class for all servers."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field

import httpx

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import ModelSpec


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
    ttft_s: float
    itl_ms: list[float] = field(default_factory=list)
    total_time_s: float = 0.0


def build_client(spec: ModelSpec) -> OpenAIGenerateClient:
    server = get_server(spec.server)
    return OpenAIGenerateClient(spec, server)


class OpenAIGenerateClient:
    def __init__(self, spec: ModelSpec, server: dict):
        self.model = spec.model_name
        host = server["host"]
        self.client = httpx.Client(base_url=host, timeout=300)
        r = self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(
                f"Server health check failed (status {r.status_code}). "
                f"Ensure server is running on {host}."
            )

    def health(self) -> bool:
        try:
            r = self.client.get("/v1/models")
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, messages: list[dict], max_tokens: int = 256,
                 temperature: float = 0.0) -> GenerateResult:
        t0 = time.perf_counter()
        r = self.client.post(
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        r.raise_for_status()
        elapsed = time.perf_counter() - t0
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return GenerateResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_time_s=elapsed,
        )

    def generate_stream(self, messages: list[dict], max_tokens: int = 256,
                        temperature: float = 0.0) -> StreamResult:
        t0 = time.perf_counter()
        ttft = 0.0
        itl_ms: list[float] = []
        chunks_text: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        last_token_time = t0
        first_token_seen = False

        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                # Extract usage if present (some servers send it in the last chunk)
                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)

                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content is None:
                    continue

                now = time.perf_counter()
                if not first_token_seen:
                    ttft = now - t0
                    first_token_seen = True
                else:
                    itl_ms.append((now - last_token_time) * 1000)
                last_token_time = now
                chunks_text.append(content)

        total_time = time.perf_counter() - t0
        return StreamResult(
            text="".join(chunks_text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens or len(chunks_text),
            ttft_s=ttft,
            itl_ms=itl_ms,
            total_time_s=total_time,
        )
```

- [ ] **Step 4: Run tests**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_generation_clients.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tqbench/benchmarks/generation/clients.py tests/benchmarks/test_generation_clients.py
git commit -m "feat: OpenAIGenerateClient with streaming TTFT/ITL measurement"
```

---

## Task 4: Speed metric aggregation

**Files:**
- Create: `tqbench/benchmarks/generation/speed_metrics.py`
- Create: `tqbench/benchmarks/generation/vram.py`
- Create: `tests/benchmarks/test_generation_speed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_generation_speed.py
from __future__ import annotations
import numpy as np
from tqbench.benchmarks.generation.speed_metrics import (
    aggregate_stream_results, aggregate_ttft_results,
)
from tqbench.benchmarks.generation.clients import StreamResult


def _stream(ttft: float, itl: list[float], comp_tokens: int = 10) -> StreamResult:
    return StreamResult(
        text="x" * comp_tokens,
        prompt_tokens=50,
        completion_tokens=comp_tokens,
        ttft_s=ttft,
        itl_ms=itl,
        total_time_s=ttft + sum(itl) / 1000,
    )


def test_aggregate_stream_results():
    results = [
        _stream(0.1, [20.0, 25.0, 30.0]),
        _stream(0.2, [22.0, 28.0, 35.0]),
        _stream(0.15, [21.0, 26.0, 32.0]),
    ]
    agg = aggregate_stream_results(results, wall_time_s=1.0)
    assert "ttft_median_s" in agg
    assert "ttft_p95_s" in agg
    assert "ttft_p99_s" in agg
    assert "itl_median_ms" in agg
    assert "itl_p95_ms" in agg
    assert "throughput_tok_s" in agg
    assert "latency_median_s" in agg
    assert "latency_p95_s" in agg
    assert agg["n_requests"] == 3
    assert agg["throughput_tok_s"] > 0
    assert agg["ttft_median_s"] == pytest.approx(0.15, abs=0.01)


def test_aggregate_ttft_results():
    ttfts = [0.5, 0.6, 0.55, 0.52, 0.58, 0.51, 0.53, 0.57, 0.54, 0.56]
    agg = aggregate_ttft_results(ttfts)
    assert "median_s" in agg
    assert "p95_s" in agg
    assert agg["n"] == 10
    assert 0.5 < agg["median_s"] < 0.6


# Need pytest for approx
import pytest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_generation_speed.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create tqbench/benchmarks/generation/speed_metrics.py**

```python
"""Aggregation functions for speed benchmark results."""
from __future__ import annotations
import numpy as np

from tqbench.benchmarks.generation.clients import StreamResult


def aggregate_stream_results(
    results: list[StreamResult],
    wall_time_s: float,
) -> dict:
    ttfts = np.array([r.ttft_s for r in results])
    all_itl = np.concatenate([np.array(r.itl_ms) for r in results if r.itl_ms])
    latencies = np.array([r.total_time_s for r in results])
    total_tokens = sum(r.completion_tokens for r in results)

    return {
        "n_requests": len(results),
        "total_output_tokens": int(total_tokens),
        "wall_time_s": wall_time_s,
        "throughput_tok_s": total_tokens / wall_time_s if wall_time_s > 0 else 0,
        "ttft_median_s": float(np.median(ttfts)),
        "ttft_p95_s": float(np.percentile(ttfts, 95)),
        "ttft_p99_s": float(np.percentile(ttfts, 99)),
        "itl_median_ms": float(np.median(all_itl)) if len(all_itl) > 0 else 0,
        "itl_p95_ms": float(np.percentile(all_itl, 95)) if len(all_itl) > 0 else 0,
        "latency_median_s": float(np.median(latencies)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
    }


def aggregate_ttft_results(ttfts: list[float]) -> dict:
    arr = np.array(ttfts)
    return {
        "n": len(arr),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "mean_s": float(np.mean(arr)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
    }
```

- [ ] **Step 4: Create tqbench/benchmarks/generation/vram.py**

```python
"""VRAM sampling via nvidia-smi in a background thread."""
from __future__ import annotations
import subprocess
import threading
import time


class VRAMSampler:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return max(self.samples) if self.samples else 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                mb = int(r.stdout.strip().splitlines()[0])
                self.samples.append(mb)
            except (ValueError, IndexError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval_s)
```

- [ ] **Step 5: Run tests**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_generation_speed.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add tqbench/benchmarks/generation/speed_metrics.py tqbench/benchmarks/generation/vram.py tests/benchmarks/test_generation_speed.py
git commit -m "feat: speed metric aggregation (TTFT, ITL, throughput) + VRAM sampler"
```

---

## Task 5: eval_quality.py — lm-evaluation-harness runner

**Files:**
- Create: `tqbench/benchmarks/generation/runners/__init__.py`
- Create: `tqbench/benchmarks/generation/runners/eval_quality.py`

- [ ] **Step 1: Create runners/__init__.py**

Empty file.

- [ ] **Step 2: Create tqbench/benchmarks/generation/runners/eval_quality.py**

```python
"""Quality evaluation via lm-evaluation-harness tinyBenchmarks.

Shells out to the lm_eval CLI rather than reimplementing eval logic.
Runs once per quality_group (fp8, q8) — not per config.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import get_candidate, quality_groups

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "raw"
TASKS = "tinyMMLU,tinyHellaSwag,tinyARC,tinyWinogrande"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_quality_eval(model_id: str) -> Path:
    spec = get_candidate(model_id)
    server = get_server(spec.server)
    host = server["host"]
    date = dt.date.today().isoformat()
    out_dir = REPORTS / f"{date}_{model_id}_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", f"model={spec.model_name},base_url={host}/v1,tokenizer_backend=huggingface",
        "--tasks", TASKS,
        "--batch_size", "1",
        "--output_path", str(out_dir),
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"lm_eval failed:\n{result.stderr}")
        raise RuntimeError(f"lm_eval exited with code {result.returncode}")

    log.info(f"Quality results written to {out_dir}")
    return out_dir


def parse_quality_results(result_dir: Path) -> dict:
    """Parse lm_eval output directory into a summary dict."""
    results_file = None
    for f in result_dir.rglob("results.json"):
        results_file = f
        break
    if not results_file:
        # Try the flat JSON format
        for f in result_dir.glob("*.json"):
            results_file = f
            break
    if not results_file:
        raise FileNotFoundError(f"No results.json found in {result_dir}")

    raw = json.loads(results_file.read_text())
    results = raw.get("results", raw)
    summary = {}
    for task_name, metrics in results.items():
        if isinstance(metrics, dict):
            acc = metrics.get("acc,none", metrics.get("acc_norm,none", metrics.get("acc")))
            if acc is not None:
                summary[task_name] = float(acc)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Specific model_id to eval")
    ap.add_argument("--group", help="Quality group to eval (fp8 or q8)")
    args = ap.parse_args()

    if args.model:
        out_dir = run_quality_eval(args.model)
        summary = parse_quality_results(out_dir)
        log.info(f"Scores: {json.dumps(summary, indent=2)}")
    elif args.group:
        groups = quality_groups()
        if args.group not in groups:
            log.error(f"Unknown group '{args.group}'. Known: {list(groups.keys())}")
            sys.exit(1)
        representative = groups[args.group][0]
        log.info(f"Running quality eval for group '{args.group}' using config '{representative.id}'")
        out_dir = run_quality_eval(representative.id)
        summary = parse_quality_results(out_dir)
        log.info(f"Scores: {json.dumps(summary, indent=2)}")
    else:
        for gname, specs in quality_groups().items():
            representative = specs[0]
            log.info(f"Group '{gname}': using '{representative.id}'")
            out_dir = run_quality_eval(representative.id)
            summary = parse_quality_results(out_dir)
            log.info(f"  Scores: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add tqbench/benchmarks/generation/runners/__init__.py tqbench/benchmarks/generation/runners/eval_quality.py
git commit -m "feat: generation quality eval via lm-evaluation-harness tinyBenchmarks"
```

---

## Task 6: eval_speed.py — async throughput/latency benchmark

**Files:**
- Create: `tqbench/benchmarks/generation/runners/eval_speed.py`

- [ ] **Step 1: Create tqbench/benchmarks/generation/runners/eval_speed.py**

```python
"""Throughput and latency benchmark at controlled concurrency levels.

For each concurrency level (1, 4, 16, 64):
- Sends 150 prompts with stream=true via async httpx
- Measures TTFT, ITL, total latency, output tokens
- Aggregates into median/p95/p99 stats
"""
from __future__ import annotations
import argparse
import asyncio
import datetime as dt
import json
import logging
import time
from pathlib import Path

import httpx
import numpy as np

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import get_candidate, load_registry
from tqbench.benchmarks.generation.speed_metrics import aggregate_stream_results
from tqbench.benchmarks.generation.vram import VRAMSampler

BENCH_ROOT = Path(__file__).resolve().parents[1]
DATA = BENCH_ROOT / "data"
REPORTS = BENCH_ROOT / "reports" / "raw"
CONCURRENCY_LEVELS = [1, 4, 16, 64]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_prompts() -> list[dict]:
    prompts: list[dict] = []
    for name in ("prompts_short.jsonl", "prompts_medium.jsonl", "prompts_long.jsonl"):
        p = DATA / name
        if not p.exists():
            log.warning(f"Missing {p}")
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


async def _stream_one(
    client: httpx.AsyncClient,
    model: str,
    prompt: dict,
) -> dict:
    """Send one streaming request and return timing metrics."""
    messages = prompt["messages"]
    max_tokens = prompt.get("max_tokens", 256)
    t0 = time.perf_counter()
    ttft = 0.0
    itl_ms: list[float] = []
    chunks: list[str] = []
    last_time = t0
    first_seen = False
    prompt_tokens = 0
    completion_tokens = 0

    async with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.0, "stream": True},
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content is None:
                continue
            now = time.perf_counter()
            if not first_seen:
                ttft = now - t0
                first_seen = True
            else:
                itl_ms.append((now - last_time) * 1000)
            last_time = now
            chunks.append(content)

    total = time.perf_counter() - t0
    return {
        "id": prompt["id"],
        "ttft_s": ttft,
        "itl_ms": itl_ms,
        "completion_tokens": completion_tokens or len(chunks),
        "prompt_tokens": prompt_tokens,
        "total_time_s": total,
        "text": "".join(chunks),
    }


async def run_concurrency_level(
    host: str, model: str, prompts: list[dict], concurrency: int,
) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=host, timeout=300, limits=limits) as client:

        async def _bounded(prompt: dict) -> dict:
            async with sem:
                return await _stream_one(client, model, prompt)

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(_bounded(p)) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.perf_counter() - t0

    valid = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        log.warning(f"  {len(errors)} requests failed")
    return valid, wall_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=CONCURRENCY_LEVELS)
    args = ap.parse_args()

    spec = get_candidate(args.model)
    server = get_server(spec.server)
    host = server["host"]

    prompts = _load_prompts()
    if not prompts:
        log.error("No prompts found in data/. Generate them first.")
        return
    log.info(f"Loaded {len(prompts)} prompts")

    vram = VRAMSampler()
    vram.start()

    all_results: dict[int, dict] = {}
    for conc in args.concurrency:
        log.info(f"Concurrency {conc}...")
        raw, wall = asyncio.run(run_concurrency_level(host, spec.model_name, prompts, conc))

        from tqbench.benchmarks.generation.clients import StreamResult
        stream_results = [
            StreamResult(
                text=r["text"], prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                ttft_s=r["ttft_s"], itl_ms=r["itl_ms"], total_time_s=r["total_time_s"],
            )
            for r in raw
        ]
        agg = aggregate_stream_results(stream_results, wall)
        all_results[conc] = agg
        log.info(f"  throughput={agg['throughput_tok_s']:.1f} tok/s  "
                 f"ttft_p50={agg['ttft_median_s']*1000:.0f}ms  "
                 f"itl_p50={agg['itl_median_ms']:.1f}ms")

    peak_vram_mb = vram.stop()

    date = dt.date.today().isoformat()
    out = {
        "model_id": spec.id,
        "server": spec.server,
        "spec_decode": spec.spec_decode,
        "spec_prefill": spec.spec_prefill,
        "peak_vram_mb": peak_vram_mb,
        "by_concurrency": {str(k): v for k, v in all_results.items()},
    }
    out_path = REPORTS / f"{date}_{spec.id}_speed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add tqbench/benchmarks/generation/runners/eval_speed.py
git commit -m "feat: async generation speed benchmark with concurrency levels"
```

---

## Task 7: eval_ttft_longctx.py — long-context TTFT isolation test

**Files:**
- Create: `tqbench/benchmarks/generation/runners/eval_ttft_longctx.py`

- [ ] **Step 1: Create tqbench/benchmarks/generation/runners/eval_ttft_longctx.py**

```python
"""Long-context TTFT isolation test.

Measures prefill time at 1K, 4K, 8K, 32K, 128K input tokens.
Sends max_tokens=1 to isolate prefill from decode.
Concurrency=1 to avoid batching effects.

This is where PFlash's sublinear prefill should be visible.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import httpx

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import get_candidate, load_registry
from tqbench.benchmarks.generation.speed_metrics import aggregate_ttft_results
from tqbench.benchmarks.generation.vram import VRAMSampler

BENCH_ROOT = Path(__file__).resolve().parents[1]
DATA = BENCH_ROOT / "data"
REPORTS = BENCH_ROOT / "reports" / "raw"
CONTEXT_LENGTHS = [1024, 4096, 8192, 32768, 131072]
REPS_PER_LENGTH = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_longctx_prompts() -> dict[int, list[dict]]:
    """Load prompts grouped by target_input_tokens."""
    p = DATA / "prompts_longctx.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Generate long-context prompts first.")
    by_length: dict[int, list[dict]] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        target = entry.get("target_input_tokens", 0)
        by_length.setdefault(target, []).append(entry)
    return by_length


def _measure_ttft(client: httpx.Client, model: str, prompt: dict) -> float:
    """Send a streaming request with max_tokens=1 and return TTFT in seconds."""
    messages = prompt["messages"]
    t0 = time.perf_counter()
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": model, "messages": messages,
              "max_tokens": 1, "temperature": 0.0, "stream": True},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content") is not None:
                return time.perf_counter() - t0
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=CONTEXT_LENGTHS)
    args = ap.parse_args()

    spec = get_candidate(args.model)
    server = get_server(spec.server)
    host = server["host"]

    prompts_by_length = _load_longctx_prompts()
    client = httpx.Client(base_url=host, timeout=600)

    # Verify server is up
    r = client.get("/v1/models")
    if r.status_code != 200:
        raise RuntimeError(f"Server not reachable at {host}")

    vram = VRAMSampler()
    vram.start()

    results_by_length: dict[str, dict] = {}
    for ctx_len in args.lengths:
        available = prompts_by_length.get(ctx_len, [])
        if not available:
            log.warning(f"No prompts for {ctx_len} tokens, skipping")
            continue
        reps = min(REPS_PER_LENGTH, len(available))
        log.info(f"Context {ctx_len:,} tokens — {reps} reps...")

        ttfts: list[float] = []
        for i in range(reps):
            prompt = available[i % len(available)]
            ttft = _measure_ttft(client, spec.model_name, prompt)
            ttfts.append(ttft)
            log.info(f"  rep {i+1}/{reps}: TTFT={ttft*1000:.0f}ms")

        agg = aggregate_ttft_results(ttfts)
        results_by_length[str(ctx_len)] = agg
        log.info(f"  median={agg['median_s']*1000:.0f}ms  p95={agg['p95_s']*1000:.0f}ms")

    peak_vram_mb = vram.stop()
    client.close()

    date = dt.date.today().isoformat()
    out = {
        "model_id": spec.id,
        "server": spec.server,
        "spec_decode": spec.spec_decode,
        "spec_prefill": spec.spec_prefill,
        "peak_vram_mb": peak_vram_mb,
        "by_context_length": results_by_length,
    }
    out_path = REPORTS / f"{date}_{spec.id}_ttft_longctx.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add tqbench/benchmarks/generation/runners/eval_ttft_longctx.py
git commit -m "feat: long-context TTFT isolation test (1K→128K tokens)"
```

---

## Task 8: run_all.py orchestrator

**Files:**
- Create: `tqbench/benchmarks/generation/runners/run_all.py`

- [ ] **Step 1: Create tqbench/benchmarks/generation/runners/run_all.py**

```python
"""Orchestrator: quality → speed → ttft → report."""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from pathlib import Path

from tqbench.benchmarks.generation.models import load_registry, quality_groups

BENCH_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _run(*cmd: str) -> int:
    log.info(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=BENCH_ROOT).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="Subset of model_ids")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-ttft", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

    models = args.models or [c.id for c in load_registry()]
    log.info(f"Pipeline begins. Models: {models}")

    # Phase 1: Quality (once per quality_group)
    if not args.skip_quality:
        for gname in quality_groups():
            _run(sys.executable, "-m",
                 "tqbench.benchmarks.generation.runners.eval_quality",
                 "--group", gname)

    # Phase 2: Speed (every config)
    if not args.skip_speed:
        for m in models:
            _run(sys.executable, "-m",
                 "tqbench.benchmarks.generation.runners.eval_speed",
                 "--model", m)

    # Phase 3: Long-context TTFT (every config)
    if not args.skip_ttft:
        for m in models:
            _run(sys.executable, "-m",
                 "tqbench.benchmarks.generation.runners.eval_ttft_longctx",
                 "--model", m)

    # Phase 4: Report
    if not args.skip_report:
        _run(sys.executable, "-m",
             "tqbench.benchmarks.generation.report")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add tqbench/benchmarks/generation/runners/run_all.py
git commit -m "feat: generation benchmark orchestrator (quality → speed → ttft → report)"
```

---

## Task 9: Report builder + template

**Files:**
- Create: `tqbench/benchmarks/generation/templates/report.html.j2`
- Create: `tqbench/benchmarks/generation/report.py`

- [ ] **Step 1: Create tqbench/benchmarks/generation/templates/report.html.j2**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Generation Benchmark — {{ run_date }}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --accent: #8abeb7; --border: #5f87ff; --bg: #1a1a1f; --bg-card: #25252b;
    --text: #e5e5e7; --muted: #808080; --success: #b5bd68; --error: #cc6666;
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, system-ui, sans-serif;
         line-height: 1.6; max-width: 1240px; margin: 0 auto; padding: 2em; }
  h1 { color: var(--accent); border-bottom: 2px solid var(--border); padding-bottom: 0.5em; }
  h2 { color: var(--accent); margin-top: 2em; border-bottom: 1px solid #333; padding-bottom: 0.25em; }
  table { border-collapse: collapse; width: 100%; background: var(--bg-card); margin: 0.5em 0 1em; }
  th, td { padding: 0.55em 0.9em; border-bottom: 1px solid #333; text-align: left; }
  th { background: rgba(95, 135, 255, 0.1); color: var(--accent); font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .chart-container { background: var(--bg-card); border-radius: 8px; padding: 1em; margin: 1em 0; }
  .chart-container.empty { padding: 1.4em; color: var(--muted); font-style: italic; text-align: center; }
  .meta { color: var(--muted); font-size: 0.9em; }
  code { background: rgba(128,128,128,0.15); padding: 0.1em 0.4em; border-radius: 3px; font-size: 0.9em; }
  .pass { color: var(--success); } .fail { color: var(--error); }
</style>
</head>
<body>
<h1>Generation Benchmark Report</h1>
<p class="meta"><strong>Run date:</strong> {{ run_date }} &middot; <strong>Configs:</strong> {{ n_configs }} &middot; <strong>Model:</strong> <code>Qwen3.5-9B</code></p>

<h2>1. Executive Summary</h2>
{% if quality_summary %}
<h3>Quality (tinyBenchmarks)</h3>
<table>
<tr><th>Group</th><th>tinyMMLU</th><th>tinyHellaSwag</th><th>tinyARC</th><th>tinyWinogrande</th><th>Status</th></tr>
{% for g in quality_summary %}
<tr>
  <td><code>{{ g.group }}</code></td>
  <td class="num">{{ "%.1f"|format(g.tinyMMLU * 100) }}%</td>
  <td class="num">{{ "%.1f"|format(g.tinyHellaSwag * 100) }}%</td>
  <td class="num">{{ "%.1f"|format(g.tinyARC * 100) }}%</td>
  <td class="num">{{ "%.1f"|format(g.tinyWinogrande * 100) }}%</td>
  <td class="{{ 'pass' if g.pass else 'fail' }}">{{ 'PASS' if g.pass else 'FAIL' }}</td>
</tr>
{% endfor %}
</table>
{% else %}
<div class="chart-container empty">No quality data.</div>
{% endif %}

<h2>2. Configuration Matrix</h2>
<table>
<tr><th>Config</th><th>Server</th><th>Prefill</th><th>Decode</th></tr>
{% for c in configs %}
<tr>
  <td><code>{{ c.id }}</code></td>
  <td>{{ c.server }}</td>
  <td>{{ c.spec_prefill or 'standard' }}</td>
  <td>{{ c.spec_decode or 'standard' }}</td>
</tr>
{% endfor %}
</table>

<h2>3. Throughput vs Concurrency</h2>
{% if throughput_chart %}<div class="chart-container">{{ throughput_chart | safe }}</div>{% else %}<div class="chart-container empty">No speed data.</div>{% endif %}

<h2>4. TTFT by Concurrency</h2>
{% if ttft_chart %}<div class="chart-container">{{ ttft_chart | safe }}</div>{% else %}<div class="chart-container empty">No TTFT data.</div>{% endif %}

<h2>5. ITL Distribution</h2>
{% if itl_chart %}<div class="chart-container">{{ itl_chart | safe }}</div>{% else %}<div class="chart-container empty">No ITL data.</div>{% endif %}

<h2>6. Long-Context TTFT (1K → 128K)</h2>
{% if longctx_chart %}<div class="chart-container">{{ longctx_chart | safe }}</div>{% else %}<div class="chart-container empty">No long-context TTFT data.</div>{% endif %}

<h2>7. VRAM Usage</h2>
{% if vram_chart %}<div class="chart-container">{{ vram_chart | safe }}</div>{% else %}<div class="chart-container empty">No VRAM data.</div>{% endif %}

<h2>8. Speculative Decoding Comparison</h2>
{% if spec_decode_chart %}<div class="chart-container">{{ spec_decode_chart | safe }}</div>{% else %}<div class="chart-container empty">No spec decode comparison data.</div>{% endif %}

<h2>9. Latency Table</h2>
{% if latency_rows %}
<table>
<tr><th>Config</th><th class="num">TTFT p50 (ms)</th><th class="num">TTFT p95 (ms)</th><th class="num">ITL p50 (ms)</th><th class="num">ITL p95 (ms)</th><th class="num">Latency p50 (s)</th><th class="num">Latency p95 (s)</th></tr>
{% for r in latency_rows %}
<tr>
  <td><code>{{ r.id }}</code></td>
  <td class="num">{{ "%.0f"|format(r.ttft_p50_ms) }}</td>
  <td class="num">{{ "%.0f"|format(r.ttft_p95_ms) }}</td>
  <td class="num">{{ "%.1f"|format(r.itl_p50_ms) }}</td>
  <td class="num">{{ "%.1f"|format(r.itl_p95_ms) }}</td>
  <td class="num">{{ "%.2f"|format(r.lat_p50_s) }}</td>
  <td class="num">{{ "%.2f"|format(r.lat_p95_s) }}</td>
</tr>
{% endfor %}
</table>
{% else %}
<div class="chart-container empty">No latency data.</div>
{% endif %}

<h2>10. Methodology</h2>
<ul class="meta">
  <li>Prompt sets: short (&lt;100 tok), medium (100-500 tok), long (500-2000 tok), longctx (1K-128K tok)</li>
  <li>Temperature: 0.0 (deterministic)</li>
  <li>Concurrency levels: 1, 4, 16, 64</li>
  <li>Quality: tinyBenchmarks via lm-evaluation-harness (tinyMMLU, tinyHellaSwag, tinyARC, tinyWinogrande)</li>
  <li>VRAM sampled every 1s via nvidia-smi</li>
</ul>

<p class="meta" style="margin-top:3em; text-align:center;">Generated {{ run_date }}</p>
</body>
</html>
```

- [ ] **Step 2: Create tqbench/benchmarks/generation/report.py**

```python
"""HTML comparison report builder for the generation benchmark."""
from __future__ import annotations
import datetime as dt
import json
import logging
from pathlib import Path

import jinja2

from tqbench.benchmarks.generation.models import load_registry

BENCH_ROOT = Path(__file__).resolve().parent
REPORTS = BENCH_ROOT / "reports"
RAW = REPORTS / "raw"
TEMPLATES = BENCH_ROOT / "templates"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_speed_data() -> dict[str, dict]:
    out = {}
    for f in RAW.glob("*_speed.json"):
        data = json.loads(f.read_text())
        out[data["model_id"]] = data
    return out


def _load_ttft_longctx_data() -> dict[str, dict]:
    out = {}
    for f in RAW.glob("*_ttft_longctx.json"):
        data = json.loads(f.read_text())
        out[data["model_id"]] = data
    return out


def _load_quality_data() -> dict[str, dict]:
    out = {}
    for d in RAW.iterdir():
        if d.is_dir() and "_quality" in d.name:
            for f in d.rglob("results.json"):
                data = json.loads(f.read_text())
                out[d.name] = data
    return out


def main() -> None:
    registry = load_registry()
    date = dt.date.today().isoformat()

    configs = [
        {"id": c.id, "server": c.server,
         "spec_decode": c.spec_decode, "spec_prefill": c.spec_prefill}
        for c in registry
    ]

    speed_data = _load_speed_data()
    ttft_data = _load_ttft_longctx_data()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES)),
        autoescape=False,
    )
    tmpl = env.get_template("report.html.j2")

    html = tmpl.render(
        run_date=date,
        n_configs=len(registry),
        configs=configs,
        quality_summary=None,
        throughput_chart=None,
        ttft_chart=None,
        itl_chart=None,
        longctx_chart=None,
        vram_chart=None,
        spec_decode_chart=None,
        latency_rows=None,
    )

    out_path = REPORTS / f"{date}_generation_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
```

Note: The report builder is scaffolded with `None` chart placeholders. Charts will be populated incrementally as benchmark data becomes available — the template gracefully shows "No data" for missing sections. Plotly chart generation is added when we have actual benchmark results to render.

- [ ] **Step 3: Commit**

```bash
git add tqbench/benchmarks/generation/templates/report.html.j2 tqbench/benchmarks/generation/report.py
git commit -m "feat: generation benchmark HTML report builder + template"
```

---

## Task 10: .gitignore + data stubs + integration test

**Files:**
- Create: `tqbench/benchmarks/generation/.gitignore`
- Create: `tqbench/benchmarks/generation/data/.gitkeep`
- Create: `tqbench/benchmarks/generation/reports/.gitkeep`
- Create: `tests/benchmarks/test_generation_integration.py`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Generated output
reports/raw/
reports/*.html
data/prompts_short.jsonl
data/prompts_medium.jsonl
data/prompts_long.jsonl
data/prompts_longctx.jsonl
```

- [ ] **Step 2: Create directory stubs**

```bash
mkdir -p tqbench/benchmarks/generation/data
touch tqbench/benchmarks/generation/data/.gitkeep
mkdir -p tqbench/benchmarks/generation/reports/raw
touch tqbench/benchmarks/generation/reports/.gitkeep
```

- [ ] **Step 3: Write integration test**

```python
# tests/benchmarks/test_generation_integration.py
"""Smoke test the generation benchmark import chain."""
from __future__ import annotations


def test_import_generation_manifest():
    from tqbench.benchmarks.generation import MANIFEST
    assert MANIFEST["name"] == "generation"
    assert MANIFEST["entry"]


def test_import_models():
    from tqbench.benchmarks.generation.models import (
        load_registry, get_candidate, quality_groups, ModelSpec,
    )
    reg = load_registry()
    assert len(reg) == 10


def test_import_clients():
    from tqbench.benchmarks.generation.clients import (
        OpenAIGenerateClient, GenerateResult, StreamResult, build_client,
    )


def test_import_speed_metrics():
    from tqbench.benchmarks.generation.speed_metrics import (
        aggregate_stream_results, aggregate_ttft_results,
    )


def test_import_vram():
    from tqbench.benchmarks.generation.vram import VRAMSampler


def test_discovery_finds_generation():
    from tqbench.benchmarks import discover_benchmarks
    benchmarks = discover_benchmarks()
    assert "generation" in benchmarks
    assert "embeddings" in benchmarks


def test_cli_list_shows_generation(capsys):
    from tqbench.cli import cmd_list
    from argparse import Namespace
    cmd_list(Namespace())
    captured = capsys.readouterr()
    assert "generation" in captured.out
    assert "embeddings" in captured.out
```

- [ ] **Step 4: Run full test suite**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/ -v --tb=short`
Expected: All tests pass (existing embeddings tests + new generation tests)

- [ ] **Step 5: Commit**

```bash
git add tqbench/benchmarks/generation/.gitignore tqbench/benchmarks/generation/data/.gitkeep tqbench/benchmarks/generation/reports/.gitkeep tests/benchmarks/test_generation_integration.py
git commit -m "test: generation benchmark integration tests + gitignore"
```

---

## Summary

| Task | What it delivers | Tests |
|---|---|---|
| 1 | New servers in servers.yaml + generation deps in pyproject.toml | existing config tests |
| 2 | MANIFEST + models.yaml (10 configs) + ModelSpec + registry + quality_groups | 7 tests |
| 3 | OpenAIGenerateClient (generate, generate_stream, health) + build_client | 4 tests |
| 4 | Speed metric aggregation (TTFT/ITL/throughput stats) + VRAM sampler | 2 tests |
| 5 | eval_quality.py — lm_eval CLI wrapper | — |
| 6 | eval_speed.py — async throughput/latency at concurrency 1/4/16/64 | — |
| 7 | eval_ttft_longctx.py — TTFT at 1K/4K/8K/32K/128K tokens | — |
| 8 | run_all.py orchestrator | — |
| 9 | HTML report builder + Jinja2 template | — |
| 10 | .gitignore + data stubs + integration tests | 7 tests |

10 tasks, ~20 tests total. The generation benchmark is fully isolated — `tqbench/benchmarks/embeddings/` is untouched.
