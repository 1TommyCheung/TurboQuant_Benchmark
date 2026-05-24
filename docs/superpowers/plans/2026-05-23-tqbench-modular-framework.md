# tqbench Modular Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a modular `tqbench/` package where the only shared artifact is `config/servers.yaml` (server deployment addresses). Each benchmark is a fully isolated sub-package with its own clients, models, eval, runners, and reports. Port the existing embeddings benchmark from `bench_embeddings_turbo/` as the first benchmark.

**Architecture:** Shared server config (~15 lines of YAML loader) + isolated benchmark sub-packages. No shared Backend protocol, no shared client library, no shared common/ utilities. Each benchmark owns all its code. Discovery via `MANIFEST` dict in each benchmark's `__init__.py`.

**Tech Stack:** Python 3.11+, PyYAML, httpx, lancedb, duckdb, numpy, pandas, pytest, Jinja2, Plotly

**Spec:** `docs/superpowers/specs/2026-05-23-tqbench-modular-design.md`

---

## File Map

### Shared (config + CLI + discovery)

| File | Responsibility |
|---|---|
| `tqbench/__init__.py` | Package marker, version |
| `tqbench/config/__init__.py` | `load_servers()`, `get_server()` — reads servers.yaml |
| `tqbench/config/servers.yaml` | Server deployment addresses (type + host) |
| `tqbench/benchmarks/__init__.py` | `discover_benchmarks()` — scans sub-packages for MANIFEST |
| `tqbench/cli.py` | CLI: `run`, `list`, `servers` commands |
| `tqbench/__main__.py` | `python -m tqbench` entry point |
| `pyproject.toml` | Package definition |

### Embeddings benchmark (ported from bench_embeddings_turbo/)

| File | Ported from | Responsibility |
|---|---|---|
| `tqbench/benchmarks/embeddings/__init__.py` | new | MANIFEST dict |
| `tqbench/benchmarks/embeddings/models.yaml` | `config/models.yaml` | Model identity + server refs (no host/port) |
| `tqbench/benchmarks/embeddings/models.py` | `src/bench/models.py` | ModelSpec dataclass, registry, `load_client()` dispatch |
| `tqbench/benchmarks/embeddings/clients.py` | `src/bench/models.py` | GeminiEmbedClient, HFEmbedClient, OllamaEmbedClient, VLLMEmbedClient, LlamaCppEmbedClient |
| `tqbench/benchmarks/embeddings/eval/__init__.py` | new | Package marker |
| `tqbench/benchmarks/embeddings/eval/metrics.py` | `src/bench/metrics.py` | recall@k, MRR, NDCG, bootstrap CI |
| `tqbench/benchmarks/embeddings/eval/sampling.py` | `src/bench/sampling.py` | Stratified 50K sampler |
| `tqbench/benchmarks/embeddings/eval/perturbations.py` | `src/bench/perturbations.py` | Typo, party abbrev, date fuzz, code-switch |
| `tqbench/benchmarks/embeddings/eval/scoring.py` | `src/bench/scoring.py` | Weighted scoring + vetoes + verdict |
| `tqbench/benchmarks/embeddings/eval/schemas.py` | `src/bench/schemas.py` | ChunkRecord pydantic model |
| `tqbench/benchmarks/embeddings/eval/leakage.py` | `src/bench/leakage.py` | n-gram overlap + cosine leakage filter |
| `tqbench/benchmarks/embeddings/eval/source_weights.py` | `src/bench/source_weights.py` | Source-type weights |
| `tqbench/benchmarks/embeddings/eval/stack.py` | `src/bench/stack.py` | vector_only, bm25, rrf, hybrid retrieve |
| `tqbench/benchmarks/embeddings/eval/io_lance.py` | `src/bench/io_lance.py` | LanceDB read/write helpers |
| `tqbench/benchmarks/embeddings/eval/snapshot.py` | `src/bench/snapshot.py` | Frozen snapshot paths |
| `tqbench/benchmarks/embeddings/eval/pool_judge.py` | `src/bench/pool_judge.py` | LLM graded relevance judging |
| `tqbench/benchmarks/embeddings/runners/__init__.py` | new | Package marker |
| `tqbench/benchmarks/embeddings/runners/run_all.py` | `runners/run_all.py` | Orchestrator |
| `tqbench/benchmarks/embeddings/runners/embed_corpus.py` | `runners/embed_corpus.py` | Phase 1 embed |
| `tqbench/benchmarks/embeddings/runners/eval_quality.py` | `runners/eval_quality.py` | Phase 1 quality eval |
| `tqbench/benchmarks/embeddings/runners/eval_speed.py` | `runners/eval_speed.py` | Phase 2 speed eval |
| `tqbench/benchmarks/embeddings/runners/build_corpus_sample.py` | `runners/build_corpus_sample.py` | Corpus sampler |
| `tqbench/benchmarks/embeddings/runners/build_layer1.py` | `runners/build_layer1.py` | Layer 1 pool-and-judge |
| `tqbench/benchmarks/embeddings/runners/build_layer2.py` | `runners/build_layer2.py` | Layer 2a perturbed queries |
| `tqbench/benchmarks/embeddings/runners/build_layer2b.py` | `runners/build_layer2b.py` | Layer 2b synthetic queries |
| `tqbench/benchmarks/embeddings/runners/build_layer2b_via_agent.py` | `runners/build_layer2b_via_agent.py` | Layer 2b via agent |
| `tqbench/benchmarks/embeddings/runners/build_adversarial.py` | `runners/build_adversarial.py` | Adversarial set |
| `tqbench/benchmarks/embeddings/runners/build_dirty.py` | `runners/build_dirty.py` | Dirty/noisy queries |
| `tqbench/benchmarks/embeddings/runners/extract_session_queries.py` | `runners/extract_session_queries.py` | Session query extraction |
| `tqbench/benchmarks/embeddings/runners/extract_speed_phase1.py` | `runners/extract_speed_phase1.py` | Phase 1 speed extraction |
| `tqbench/benchmarks/embeddings/runners/smoke_test_vllm.py` | `runners/smoke_test_vllm.py` | vLLM smoke test |
| `tqbench/benchmarks/embeddings/report.py` | `runners/build_report.py` | Main HTML report builder |
| `tqbench/benchmarks/embeddings/report_turboquant_compare.py` | `runners/build_turboquant_compare_report.py` | TQ compare report |
| `tqbench/benchmarks/embeddings/report_turboquant_matrix.py` | `runners/build_turboquant_matrix_report.py` | TQ matrix report |
| `tqbench/benchmarks/embeddings/report_fp8_cross_query.py` | `runners/build_fp8_cross_query_report.py` | FP8 cross-query report |
| `tqbench/benchmarks/embeddings/report_pool_judge.py` | `runners/build_pool_judge_report.py` | Pool-judge report |
| `tqbench/benchmarks/embeddings/templates/report.html.j2` | `templates/report.html.j2` | Jinja2 HTML template |

### Tests

| File | Tests for |
|---|---|
| `tests/__init__.py` | Package marker |
| `tests/test_config.py` | `load_servers()`, `get_server()` |
| `tests/test_discovery.py` | `discover_benchmarks()` |
| `tests/benchmarks/__init__.py` | Package marker |
| `tests/benchmarks/test_embeddings_models.py` | ModelSpec, registry, load_client dispatch |
| `tests/benchmarks/test_embeddings_clients.py` | Client classes (mocked HTTP) |
| `tests/benchmarks/test_embeddings_metrics.py` | recall@k, MRR, NDCG, bootstrap CI |
| `tests/benchmarks/test_embeddings_scoring.py` | Weighted scoring, vetoes, verdict |
| `tests/benchmarks/test_embeddings_eval.py` | leakage, perturbations, sampling, stack |

---

## Task 1: pyproject.toml + package skeleton

**Files:**
- Create: `tqbench/__init__.py`
- Create: `tqbench/__main__.py`
- Create: `tqbench/py.typed`
- Create: `pyproject.toml` — at `tqbench/pyproject.toml` (sibling to existing bench_embeddings_turbo/)
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "tqbench"
version = "0.1.0"
description = "Modular inference benchmark framework — server config shared, benchmarks isolated"
requires-python = ">=3.11"
dependencies = [
    "pyyaml>=6",
    "httpx>=0.27",
]

[project.optional-dependencies]
embeddings = [
    "lancedb>=0.13",
    "duckdb>=1.0",
    "pyarrow>=15",
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "plotly>=5.20",
    "jinja2>=3.1",
    "tqdm>=4.66",
]
embeddings-hf = [
    "sentence-transformers>=3.0",
    "transformers>=4.46",
    "bitsandbytes>=0.43",
    "torch>=2.4",
]
embeddings-judge = ["litellm>=1.50"]
test = ["pytest>=8", "pytest-mock>=3.12"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["tqbench*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create tqbench/__init__.py**

```python
"""tqbench — modular inference benchmark framework."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Create tqbench/__main__.py**

```python
from tqbench.cli import main

main()
```

- [ ] **Step 4: Create tqbench/py.typed**

Empty file (marker for PEP 561).

- [ ] **Step 5: Create tests/__init__.py**

Empty file.

- [ ] **Step 6: Commit**

```bash
git add tqbench/__init__.py tqbench/__main__.py tqbench/py.typed tqbench/pyproject.toml tests/__init__.py
git commit -m "feat: tqbench package skeleton and pyproject.toml"
```

---

## Task 2: Shared server config

**Files:**
- Create: `tqbench/config/__init__.py`
- Create: `tqbench/config/servers.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from __future__ import annotations
import pytest
from tqbench.config import load_servers, get_server


def test_load_servers_returns_dict():
    servers = load_servers()
    assert isinstance(servers, dict)
    assert len(servers) > 0


def test_get_server_known():
    server = get_server("ollama-local")
    assert server["type"] == "ollama"
    assert "host" in server


def test_get_server_unknown_raises():
    with pytest.raises(KeyError, match="no-such-server"):
        get_server("no-such-server")


def test_all_servers_have_type_and_host():
    for name, server in load_servers().items():
        assert "type" in server, f"Server '{name}' missing 'type'"
        if server["type"] != "hf":
            assert "host" in server, f"Server '{name}' missing 'host'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tqbench.config'`

- [ ] **Step 3: Create tqbench/config/servers.yaml**

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

  gemini:
    type: gemini_api

  hf-local:
    type: hf
    device: cuda
```

- [ ] **Step 4: Create tqbench/config/__init__.py**

```python
"""Shared server deployment config — the only shared artifact across benchmarks."""
from __future__ import annotations
from pathlib import Path
import yaml

SERVERS_PATH = Path(__file__).parent / "servers.yaml"


def load_servers() -> dict[str, dict]:
    raw = yaml.safe_load(SERVERS_PATH.read_text())
    return raw["servers"]


def get_server(name: str) -> dict:
    servers = load_servers()
    if name not in servers:
        raise KeyError(
            f"Unknown server '{name}'. Known: {sorted(servers.keys())}"
        )
    return servers[name]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add tqbench/config/__init__.py tqbench/config/servers.yaml tests/test_config.py
git commit -m "feat: shared server config loader and servers.yaml"
```

---

## Task 3: Benchmark discovery

**Files:**
- Create: `tqbench/benchmarks/__init__.py`
- Create: `tqbench/benchmarks/embeddings/__init__.py`
- Create: `tests/test_discovery.py`
- Create: `tests/benchmarks/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_discovery.py
from __future__ import annotations
from tqbench.benchmarks import discover_benchmarks


def test_discover_finds_embeddings():
    benchmarks = discover_benchmarks()
    assert "embeddings" in benchmarks
    assert benchmarks["embeddings"]["description"]
    assert benchmarks["embeddings"]["entry"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/test_discovery.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create tqbench/benchmarks/embeddings/__init__.py**

```python
"""Embeddings benchmark — embedding model quality and speed bake-off."""

MANIFEST = {
    "name": "embeddings",
    "description": "Embedding model quality and speed bake-off",
    "entry": "tqbench.benchmarks.embeddings.runners.run_all:main",
}
```

- [ ] **Step 4: Create tqbench/benchmarks/__init__.py**

```python
"""Benchmark discovery — scans sub-packages for MANIFEST dicts."""
from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path


def discover_benchmarks() -> dict[str, dict]:
    benchmarks: dict[str, dict] = {}
    pkg_path = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_path)]):
        if not info.ispkg:
            continue
        try:
            mod = importlib.import_module(f"tqbench.benchmarks.{info.name}")
            manifest = getattr(mod, "MANIFEST", None)
            if manifest and isinstance(manifest, dict) and "name" in manifest:
                benchmarks[manifest["name"]] = manifest
        except ImportError:
            continue
    return benchmarks
```

- [ ] **Step 5: Create tests/benchmarks/__init__.py**

Empty file.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/test_discovery.py -v`
Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add tqbench/benchmarks/__init__.py tqbench/benchmarks/embeddings/__init__.py tests/test_discovery.py tests/benchmarks/__init__.py
git commit -m "feat: benchmark discovery via MANIFEST + embeddings manifest"
```

---

## Task 4: CLI

**Files:**
- Create: `tqbench/cli.py`

- [ ] **Step 1: Create tqbench/cli.py**

```python
"""CLI entry point: python -m tqbench <command>."""
from __future__ import annotations
import argparse
import importlib
import sys

from tqbench.benchmarks import discover_benchmarks
from tqbench.config import load_servers


def cmd_list(args: argparse.Namespace) -> None:
    benchmarks = discover_benchmarks()
    if not benchmarks:
        print("No benchmarks found.")
        return
    for name, manifest in sorted(benchmarks.items()):
        print(f"  {name:20s}  {manifest.get('description', '')}")


def cmd_servers(args: argparse.Namespace) -> None:
    servers = load_servers()
    target = getattr(args, "server_name", None)
    if target:
        if target not in servers:
            print(f"Unknown server: {target}")
            sys.exit(1)
        s = servers[target]
        print(f"  {target}: type={s['type']} host={s.get('host', 'n/a')}")
        return
    for name, s in sorted(servers.items()):
        print(f"  {name:20s}  type={s['type']:12s}  host={s.get('host', 'n/a')}")


def cmd_run(args: argparse.Namespace) -> None:
    benchmarks = discover_benchmarks()
    if args.benchmark not in benchmarks:
        print(f"Unknown benchmark: {args.benchmark}")
        print(f"Available: {', '.join(sorted(benchmarks.keys()))}")
        sys.exit(1)
    manifest = benchmarks[args.benchmark]
    module_path, func_name = manifest["entry"].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    entry_fn = getattr(mod, func_name)
    entry_fn()


def main() -> None:
    parser = argparse.ArgumentParser(prog="tqbench", description="Modular inference benchmark framework")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available benchmarks")

    srv = sub.add_parser("servers", help="Show server config")
    srv.add_argument("server_name", nargs="?", help="Specific server to inspect")

    run_p = sub.add_parser("run", help="Run a benchmark")
    run_p.add_argument("benchmark", help="Benchmark name (e.g. embeddings)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    {"list": cmd_list, "servers": cmd_servers, "run": cmd_run}[args.command](args)
```

- [ ] **Step 2: Smoke test the CLI**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m tqbench list`
Expected: output showing `embeddings` with its description

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m tqbench servers`
Expected: output listing all servers from servers.yaml

- [ ] **Step 3: Commit**

```bash
git add tqbench/cli.py tqbench/__main__.py
git commit -m "feat: tqbench CLI with list, servers, run commands"
```

---

## Task 5: Embeddings eval modules (pure functions)

Port the pure-function eval modules from `bench_embeddings_turbo/src/bench/` into `tqbench/benchmarks/embeddings/eval/`. These have no server dependencies and can be tested in isolation.

**Files:**
- Create: `tqbench/benchmarks/embeddings/eval/__init__.py`
- Create: `tqbench/benchmarks/embeddings/eval/metrics.py`
- Create: `tqbench/benchmarks/embeddings/eval/sampling.py`
- Create: `tqbench/benchmarks/embeddings/eval/perturbations.py`
- Create: `tqbench/benchmarks/embeddings/eval/scoring.py`
- Create: `tqbench/benchmarks/embeddings/eval/schemas.py`
- Create: `tqbench/benchmarks/embeddings/eval/leakage.py`
- Create: `tqbench/benchmarks/embeddings/eval/source_weights.py`
- Create: `tqbench/benchmarks/embeddings/eval/stack.py`
- Create: `tqbench/benchmarks/embeddings/eval/io_lance.py`
- Create: `tqbench/benchmarks/embeddings/eval/snapshot.py`
- Create: `tqbench/benchmarks/embeddings/eval/pool_judge.py`
- Create: `tests/benchmarks/test_embeddings_metrics.py`
- Create: `tests/benchmarks/test_embeddings_scoring.py`
- Create: `tests/benchmarks/test_embeddings_eval.py`

- [ ] **Step 1: Create tqbench/benchmarks/embeddings/eval/__init__.py**

Empty file.

- [ ] **Step 2: Copy eval modules**

Copy each source file from `bench_embeddings_turbo/src/bench/` to `tqbench/benchmarks/embeddings/eval/`, preserving contents exactly:

| Source | Destination |
|---|---|
| `bench_embeddings_turbo/src/bench/metrics.py` | `tqbench/benchmarks/embeddings/eval/metrics.py` |
| `bench_embeddings_turbo/src/bench/sampling.py` | `tqbench/benchmarks/embeddings/eval/sampling.py` |
| `bench_embeddings_turbo/src/bench/perturbations.py` | `tqbench/benchmarks/embeddings/eval/perturbations.py` |
| `bench_embeddings_turbo/src/bench/scoring.py` | `tqbench/benchmarks/embeddings/eval/scoring.py` |
| `bench_embeddings_turbo/src/bench/schemas.py` | `tqbench/benchmarks/embeddings/eval/schemas.py` |
| `bench_embeddings_turbo/src/bench/leakage.py` | `tqbench/benchmarks/embeddings/eval/leakage.py` |
| `bench_embeddings_turbo/src/bench/source_weights.py` | `tqbench/benchmarks/embeddings/eval/source_weights.py` |
| `bench_embeddings_turbo/src/bench/stack.py` | `tqbench/benchmarks/embeddings/eval/stack.py` |
| `bench_embeddings_turbo/src/bench/pool_judge.py` | `tqbench/benchmarks/embeddings/eval/pool_judge.py` |

These files are pure functions with no internal cross-imports that need updating (they only import stdlib + numpy/pandas/pydantic/litellm).

- [ ] **Step 3: Copy io_lance.py with updated import path**

Copy `bench_embeddings_turbo/src/bench/io_lance.py` to `tqbench/benchmarks/embeddings/eval/io_lance.py`.

Update the relative import on line 11 from:
```python
from .snapshot import SNAPSHOT_CHUNKS_PARQUET, SNAPSHOT_LANCEDB_PATH
```
to:
```python
from .snapshot import SNAPSHOT_CHUNKS_PARQUET, SNAPSHOT_LANCEDB_PATH
```

No change needed — the relative import already works in the new location.

Update `BENCH_LANCEDB_ROOT` on line 15 — the directory depth has changed. In the old layout it was `parents[2]` (from `src/bench/` up to `bench_embeddings_turbo/`). In the new layout it goes from `tqbench/benchmarks/embeddings/eval/` and we want `tqbench/benchmarks/embeddings/indexes/`:

```python
BENCH_LANCEDB_ROOT = Path(__file__).resolve().parents[1] / "indexes"
```

Change `parents[2]` → `parents[1]` since we're now one level closer to the benchmark root.

- [ ] **Step 4: Copy snapshot.py unchanged**

Copy `bench_embeddings_turbo/src/bench/snapshot.py` to `tqbench/benchmarks/embeddings/eval/snapshot.py`. No changes needed — all paths are absolute Windows paths.

- [ ] **Step 5: Write test for metrics**

```python
# tests/benchmarks/test_embeddings_metrics.py
from __future__ import annotations
import numpy as np
from tqbench.benchmarks.embeddings.eval.metrics import (
    recall_at_k, mrr_at_k, ndcg_at_k, bootstrap_ci,
)


def test_recall_at_k_all_in_topk():
    ranked = ["a", "b", "c", "d", "e"]
    positives = {"a", "c"}
    assert recall_at_k(ranked, positives, k=5) == 1.0


def test_recall_at_k_partial():
    ranked = ["x", "a", "y", "z"]
    positives = {"a", "b"}
    assert recall_at_k(ranked, positives, k=4) == 0.5


def test_recall_at_k_none():
    ranked = ["x", "y", "z"]
    positives = {"a", "b"}
    assert recall_at_k(ranked, positives, k=3) == 0.0


def test_mrr_at_k_first_hit():
    ranked = ["x", "a", "b"]
    positives = {"a"}
    assert mrr_at_k(ranked, positives, k=3) == 0.5


def test_mrr_at_k_no_hit():
    ranked = ["x", "y", "z"]
    positives = {"a"}
    assert mrr_at_k(ranked, positives, k=3) == 0.0


def test_ndcg_at_k_perfect():
    ranked = ["a", "b", "c"]
    grades = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(ranked, grades, k=3) == 1.0


def test_ndcg_at_k_reverse():
    ranked = ["c", "b", "a"]
    grades = {"a": 3, "b": 2, "c": 1}
    score = ndcg_at_k(ranked, grades, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_zero_grades():
    ranked = ["a", "b"]
    grades = {"a": 0, "b": 0}
    assert ndcg_at_k(ranked, grades, k=2) == 0.0


def test_bootstrap_ci_returns_interval():
    rng = np.random.default_rng(42)
    scores = rng.normal(0.5, 0.1, size=200)
    lo, hi = bootstrap_ci(scores, n_resamples=500, seed=42)
    assert lo < hi
    assert 0.3 < lo < 0.6
    assert 0.4 < hi < 0.7
```

- [ ] **Step 6: Write test for scoring**

```python
# tests/benchmarks/test_embeddings_scoring.py
from __future__ import annotations
from tqbench.benchmarks.embeddings.eval.scoring import (
    ModelResult, apply_vetoes, weighted_total, decide,
)


def _make_result(model_id: str, dim: int = 3072, e2e_court: float = 80.0,
                 base_court: float = 80.0, **kw) -> ModelResult:
    defaults = dict(
        quality_vector_only=70.0,
        quality_end_to_end=75.0,
        long_context=60.0,
        local_control=50.0,
        e2e_recall_by_source_type={"court_doc": e2e_court, "solicitor_letter": 85.0},
        baseline_e2e_recall_by_source_type={"court_doc": base_court, "solicitor_letter": 85.0},
        dim=dim,
        baseline_dim=3072,
    )
    defaults.update(kw)
    return ModelResult(model_id=model_id, **defaults)


def test_no_veto_when_within_threshold():
    r = _make_result("qwen", e2e_court=76.0, base_court=80.0)
    assert apply_vetoes(r) == []


def test_veto_when_regression_exceeds_threshold():
    r = _make_result("qwen", e2e_court=74.0, base_court=80.0)
    vetoes = apply_vetoes(r)
    assert len(vetoes) == 1
    assert "VETO" in vetoes[0]


def test_dim_penalty_applied():
    r = _make_result("qwen", dim=4096)
    total_with_penalty = weighted_total(r)
    r2 = _make_result("qwen", dim=3072)
    total_without = weighted_total(r2)
    assert total_with_penalty == total_without - 3.0


def test_decide_stay_when_all_vetoed():
    baseline = _make_result("gemini")
    candidate = _make_result("qwen", e2e_court=60.0, base_court=80.0)
    verdict = decide(candidate, baseline, [baseline, candidate])
    assert verdict.verdict == "stay"
```

- [ ] **Step 7: Write test for eval utilities (leakage, perturbations, stack)**

```python
# tests/benchmarks/test_embeddings_eval.py
from __future__ import annotations
import numpy as np
from tqbench.benchmarks.embeddings.eval.leakage import ngram_overlap, cosine, is_leaky
from tqbench.benchmarks.embeddings.eval.perturbations import inject_typo, perturb_all
from tqbench.benchmarks.embeddings.eval.stack import rrf_fuse
from tqbench.benchmarks.embeddings.eval.source_weights import weight_for, weighted_cited_overlap


def test_ngram_overlap_identical():
    s = "the quick brown fox jumps over the lazy dog"
    assert ngram_overlap(s, s) == 1.0


def test_ngram_overlap_disjoint():
    assert ngram_overlap("alpha beta gamma delta", "one two three four") == 0.0


def test_cosine_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine(v, v) == 1.0


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine(a, b) == 0.0


def test_is_leaky_high_overlap():
    text = "the quick brown fox jumps over the lazy dog near the river"
    v = np.array([1.0, 0.0])
    assert is_leaky(text, text, v, v)


def test_inject_typo_changes_text():
    q = "what emails did tommy send in february"
    result = inject_typo(q, seed=42)
    assert result != q


def test_perturb_all_produces_variants():
    q = "What did Lee & Lee say about the custody hearing in January 2026?"
    variants = perturb_all(q, seed=42)
    assert len(variants) >= 3
    assert all(v != q for v in variants)


def test_rrf_fuse_basic():
    list1 = ["a", "b", "c"]
    list2 = ["b", "c", "d"]
    fused = rrf_fuse([list1, list2], k=3)
    assert "b" in fused
    assert len(fused) == 3


def test_weight_for_court_doc():
    assert weight_for("court_doc") == 2.0
    assert weight_for("whatsapp") == 1.0
    assert weight_for("unknown_type") == 1.0


def test_weighted_cited_overlap_all_found():
    cited = [("c1", "court_doc"), ("c2", "email")]
    returned = {"c1", "c2", "c3"}
    assert weighted_cited_overlap(cited, returned) == 1.0


def test_weighted_cited_overlap_none_found():
    cited = [("c1", "court_doc")]
    returned = {"c99"}
    assert weighted_cited_overlap(cited, returned) == 0.0
```

- [ ] **Step 8: Run all tests**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/ -v`
Expected: All tests pass

- [ ] **Step 9: Commit**

```bash
git add tqbench/benchmarks/embeddings/eval/ tests/benchmarks/test_embeddings_metrics.py tests/benchmarks/test_embeddings_scoring.py tests/benchmarks/test_embeddings_eval.py
git commit -m "feat: port embeddings eval modules (metrics, scoring, sampling, perturbations, leakage, stack)"
```

---

## Task 6: Embeddings models.yaml + ModelSpec + registry

**Files:**
- Create: `tqbench/benchmarks/embeddings/models.yaml`
- Create: `tqbench/benchmarks/embeddings/models.py`
- Create: `tests/benchmarks/test_embeddings_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_embeddings_models.py
from __future__ import annotations
import pytest
from tqbench.benchmarks.embeddings.models import load_registry, get_candidate, ModelSpec


def test_registry_loads():
    reg = load_registry()
    assert len(reg) > 0
    assert all(isinstance(m, ModelSpec) for m in reg)


def test_every_candidate_has_server_ref():
    for m in load_registry():
        assert m.server, f"Model '{m.id}' missing server reference"


def test_get_candidate_by_id():
    spec = get_candidate("qwen3-embedding-8b-q8-ollama")
    assert spec.dim == 4096
    assert spec.server == "ollama-local"


def test_get_candidate_missing_raises():
    with pytest.raises(KeyError):
        get_candidate("does-not-exist")


def test_baseline_dim():
    from tqbench.benchmarks.embeddings.models import baseline_dim
    assert baseline_dim() == 3072
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_embeddings_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create tqbench/benchmarks/embeddings/models.yaml**

This is the existing `config/models.yaml` with server fields replaced by `server:` references. Each model entry gets a `server` field pointing to a name in `config/servers.yaml`. The `kind` field is removed (server type comes from servers.yaml).

```yaml
candidates:
  - id: gemini-embedding-001
    server: gemini
    dim: 3072
    max_ctx_tokens: 2048
    precision: api
    notes: Production baseline (V1).

  - id: gemini-embedding-2
    server: gemini
    dim: 3072
    max_ctx_tokens: 32768
    precision: api
    notes: Gemini V2 — same dim as V1 (3072), 4x context window.

  - id: qwen3-embedding-8b-fp16
    server: hf-local
    hf_repo: Qwen/Qwen3-Embedding-8B
    dim: 4096
    max_ctx_tokens: 32768
    precision: fp16
    vram_estimate_gb: 16
    notes: Upper bound for the primary candidate.

  - id: qwen3-embedding-8b-int8
    server: hf-local
    hf_repo: Qwen/Qwen3-Embedding-8B
    dim: 4096
    max_ctx_tokens: 32768
    precision: int8
    quantization: bitsandbytes
    vram_estimate_gb: 9
    notes: Production candidate (matches user intent).

  - id: kalm-gemma3-12b-int8
    server: hf-local
    hf_repo: tencent/KaLM-Embedding-Gemma3-12B-2511
    dim: 3584
    max_ctx_tokens: 32768
    precision: int8
    quantization: bitsandbytes
    vram_estimate_gb: 15
    notes: Top-of-leaderboard local model that still fits.

  - id: llama-embed-nemotron-8b-int8
    server: hf-local
    hf_repo: nvidia/llama-embed-nemotron-8b
    dim: 4096
    max_ctx_tokens: 32768
    precision: fp16
    quantization: bitsandbytes
    vram_estimate_gb: 16
    notes: NVIDIA peer to Qwen3-8B.

  - id: harrier-oss-0.6b-bf16
    server: hf-local
    hf_repo: microsoft/harrier-oss-v1-0.6b
    dim: 1024
    max_ctx_tokens: 32768
    precision: bf16
    vram_estimate_gb: 2
    notes: Sleeper — MTEB avg 69.01 at 0.6B params.

  - id: qwen3-embedding-4b-fp16
    server: hf-local
    hf_repo: Qwen/Qwen3-Embedding-4B
    dim: 2560
    max_ctx_tokens: 32768
    precision: fp16
    vram_estimate_gb: 8
    notes: Right-sizer between 0.6B and 8B.

  - id: jina-v5-text-small
    server: hf-local
    hf_repo: jinaai/jina-embeddings-v5-text-small
    dim: 1024
    max_ctx_tokens: 32768
    precision: fp16
    vram_estimate_gb: 3
    notes: Jina's small text-only v5.

  - id: qwen3-embedding-8b-q8-ollama
    server: ollama-local
    ollama_model: qwen3-embedding:8b-q8_0
    dim: 4096
    max_ctx_tokens: 32768
    precision: q8_gguf
    quantization: llama_cpp_q8
    vram_estimate_gb: 9
    notes: GGUF Q8_0 via Ollama.

  - id: qwen3-embedding-8b-q8-turbo
    server: turboquant-local
    hf_repo: Qwen/Qwen3-Embedding-8B-GGUF
    llamacpp_model: qwen3-embedding-8b-q8-turbo
    dim: 4096
    max_ctx_tokens: 32768
    precision: q8_gguf
    quantization: llama_cpp_q8_turboquant
    vram_estimate_gb: 9
    notes: Official Qwen3-Embedding-8B Q8_0 GGUF served by TurboQuant.

  - id: qwen3-embedding-8b-q8-tq-turbo3
    server: turboquant-local
    hf_repo: Qwen/Qwen3-Embedding-8B-GGUF
    llamacpp_model: qwen3-embedding-8b-q8-tq-turbo3
    dim: 4096
    max_ctx_tokens: 32768
    precision: q8_gguf
    quantization: llama_cpp_q8_turboquant_turbo3_turbo3
    vram_estimate_gb: 9
    notes: TurboQuant matrix config A; symmetric turbo3 K/V.

  - id: qwen3-embedding-8b-q8-tq-turbo4
    server: turboquant-local
    hf_repo: Qwen/Qwen3-Embedding-8B-GGUF
    llamacpp_model: qwen3-embedding-8b-q8-tq-turbo4
    dim: 4096
    max_ctx_tokens: 32768
    precision: q8_gguf
    quantization: llama_cpp_q8_turboquant_turbo4_turbo4
    vram_estimate_gb: 10
    notes: TurboQuant matrix config B; symmetric turbo4 K/V.

  - id: qwen3-embedding-8b-q8-tq-q8-turbo4
    server: turboquant-local
    hf_repo: Qwen/Qwen3-Embedding-8B-GGUF
    llamacpp_model: qwen3-embedding-8b-q8-tq-q8-turbo4
    dim: 4096
    max_ctx_tokens: 32768
    precision: q8_gguf
    quantization: llama_cpp_q8_turboquant_q8_0_turbo4
    vram_estimate_gb: 11
    notes: TurboQuant matrix config C; q8_0 K with turbo4 V.

  - id: qwen3-embedding-8b-q8-tq-q8-q8
    server: turboquant-local
    hf_repo: Qwen/Qwen3-Embedding-8B-GGUF
    llamacpp_model: qwen3-embedding-8b-q8-tq-q8-q8
    dim: 4096
    max_ctx_tokens: 8192
    precision: q8_gguf
    quantization: llama_cpp_q8_q8_0_q8_0
    vram_estimate_gb: 12
    notes: llama.cpp q8_0 K/V baseline.

  - id: qwen3-embedding-8b-fp8-vllm
    server: vllm-docker
    hf_repo: maywell/Qwen3-Embedding-8B-FP8-Dynamic
    dim: 4096
    max_ctx_tokens: 32768
    precision: fp8_dynamic
    quantization: compressed_tensors
    vram_estimate_gb: 9
    notes: maywell's FP8-Dynamic via vLLM.

  - id: qwen3-embedding-8b-fp8-vllm-docker-8k
    server: vllm-docker
    hf_repo: maywell/Qwen3-Embedding-8B-FP8-Dynamic
    dim: 4096
    max_ctx_tokens: 8192
    precision: fp8_dynamic
    quantization: compressed_tensors
    vram_estimate_gb: 9
    notes: vLLM 8K apples-to-apples rerun.

  - id: qwen3-embedding-4b-fp8-vllm
    server: vllm-docker
    hf_repo: chroma-core/Qwen3-Embedding-4B-FP8-Dynamic
    dim: 2560
    max_ctx_tokens: 32768
    precision: fp8_dynamic
    quantization: compressed_tensors
    vram_estimate_gb: 5
    notes: chroma-core's FP8-Dynamic 4B.

  - id: nv-embed-v2-fp16
    server: hf-local
    hf_repo: nvidia/NV-Embed-v2
    dim: 4096
    max_ctx_tokens: 32768
    precision: fp16
    vram_estimate_gb: 16
    notes: NVIDIA NV-Embed-v2. Requires trust_remote_code.

baseline_dim: 3072
```

- [ ] **Step 4: Create tqbench/benchmarks/embeddings/models.py**

```python
"""Model registry for the embeddings benchmark.

Reads models.yaml, exposes typed accessors. Server connection details
come from tqbench.config.get_server() — this module only handles
model identity (weights, dimensions, precision).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    server: str
    dim: int
    max_ctx_tokens: int
    precision: str
    hf_repo: str | None = None
    ollama_model: str | None = None
    llamacpp_model: str | None = None
    quantization: str | None = None
    vram_estimate_gb: float | None = None
    notes: str | None = None


def load_registry() -> list[ModelSpec]:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    return [ModelSpec(**c) for c in raw["candidates"]]


def get_candidate(model_id: str) -> ModelSpec:
    for c in load_registry():
        if c.id == model_id:
            return c
    raise KeyError(model_id)


def baseline_dim() -> int:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    return int(raw["baseline_dim"])


def load_client(model_id: str):
    """Return an embedder with .encode(texts, batch_size) -> np.ndarray."""
    from tqbench.benchmarks.embeddings.clients import build_client
    spec = get_candidate(model_id)
    return build_client(spec)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_embeddings_models.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add tqbench/benchmarks/embeddings/models.yaml tqbench/benchmarks/embeddings/models.py tests/benchmarks/test_embeddings_models.py
git commit -m "feat: embeddings ModelSpec, registry, and models.yaml with server refs"
```

---

## Task 7: Embeddings client classes

**Files:**
- Create: `tqbench/benchmarks/embeddings/clients.py`
- Create: `tests/benchmarks/test_embeddings_clients.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_embeddings_clients.py
from __future__ import annotations
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from tqbench.benchmarks.embeddings.models import ModelSpec


def _spec(server: str = "ollama-local", **kw) -> ModelSpec:
    defaults = dict(id="test-model", server=server, dim=4096,
                    max_ctx_tokens=8192, precision="q8_gguf")
    defaults.update(kw)
    return ModelSpec(**defaults)


def test_build_client_dispatches_by_server_type():
    from tqbench.benchmarks.embeddings.clients import build_client
    from tqbench.config import get_server

    spec = _spec(server="ollama-local", ollama_model="test:latest")
    server_conf = get_server("ollama-local")
    assert server_conf["type"] == "ollama"


def test_ollama_encode_normalizes():
    from tqbench.benchmarks.embeddings.clients import OllamaEmbedClient

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": [[3.0, 4.0], [1.0, 0.0]]
    }
    mock_client.post.return_value = mock_response

    spec = _spec(ollama_model="test:latest")
    client = OllamaEmbedClient.__new__(OllamaEmbedClient)
    client.spec = spec
    client.client = mock_client
    client.ollama_model = "test:latest"

    result = client.encode(["hello", "world"], batch_size=8)
    assert result.shape == (2, 2)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_llamacpp_encode_normalizes():
    from tqbench.benchmarks.embeddings.clients import LlamaCppEmbedClient

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [3.0, 4.0]}, {"embedding": [1.0, 0.0]}]
    }
    mock_client.post.return_value = mock_response
    mock_client.get.return_value = MagicMock(status_code=200)

    spec = _spec(server="turboquant-local", llamacpp_model="test-model")
    client = LlamaCppEmbedClient.__new__(LlamaCppEmbedClient)
    client.spec = spec
    client.model_name = "test-model"
    client.client = mock_client

    result = client.encode(["hello", "world"], batch_size=8)
    assert result.shape == (2, 2)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_embeddings_clients.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create tqbench/benchmarks/embeddings/clients.py**

Port the five embedder classes from `bench_embeddings_turbo/src/bench/models.py`, updating them to read server host from `tqbench.config.get_server()`.

```python
"""Embedding client classes — one per server type.

Each client has .encode(texts: list[str], batch_size: int) -> np.ndarray
returning L2-normalized float32 vectors. Server connection details come
from tqbench.config.get_server(); model identity from ModelSpec.
"""
from __future__ import annotations
from tqbench.config import get_server

if __name__ != "__main__":
    from tqbench.benchmarks.embeddings.models import ModelSpec


def build_client(spec: ModelSpec):
    server = get_server(spec.server)
    server_type = server["type"]
    if server_type == "gemini_api":
        return GeminiEmbedClient(spec, server)
    if server_type == "ollama":
        return OllamaEmbedClient(spec, server)
    if server_type == "vllm":
        return VLLMEmbedClient(spec, server)
    if server_type == "llamacpp":
        return LlamaCppEmbedClient(spec, server)
    if server_type == "hf":
        return HFEmbedClient(spec, server)
    raise ValueError(f"Unknown server type '{server_type}' for server '{spec.server}'")


class GeminiEmbedClient:
    def __init__(self, spec: ModelSpec, server: dict):
        import google.generativeai as genai
        import os
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.spec = spec
        self.genai = genai

    def encode(self, texts: list[str], batch_size: int = 100):
        import numpy as np
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            out.extend(self._encode_batch_recursive(batch))
        return np.array(out, dtype="float32")

    def _encode_batch_recursive(self, batch: list[str], rate_attempt: int = 0) -> list:
        import time
        if not batch:
            return []
        try:
            resp = self.genai.embed_content(
                model=f"models/{self.spec.id}",
                content=batch,
                task_type="retrieval_document",
            )
            return list(resp["embedding"])
        except Exception as e:
            msg = str(e).lower()
            is_rate = "429" in msg or "resource exhausted" in msg or "rate" in msg or "503" in msg
            is_deadline = "deadline" in msg or "504" in msg
            if is_rate:
                if rate_attempt >= 8:
                    raise
                time.sleep(min(2.0 * (2 ** rate_attempt), 60.0))
                return self._encode_batch_recursive(batch, rate_attempt + 1)
            if is_deadline:
                if len(batch) == 1:
                    raise
                mid = len(batch) // 2
                return (
                    self._encode_batch_recursive(batch[:mid])
                    + self._encode_batch_recursive(batch[mid:])
                )
            raise


class HFEmbedClient:
    def __init__(self, spec: ModelSpec, server: dict):
        from sentence_transformers import SentenceTransformer
        import torch
        device = server.get("device", "cuda")
        kwargs: dict = {"device": device, "trust_remote_code": True}
        if spec.precision == "int8":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["model_kwargs"] = {"quantization_config": bnb_config}
        elif spec.precision == "fp16":
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        elif spec.precision == "bf16":
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        self.model = SentenceTransformer(spec.hf_repo, **kwargs)
        self.model.max_seq_length = min(spec.max_ctx_tokens, 2048)
        self.spec = spec

    def encode(self, texts: list[str], batch_size: int = 32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


class OllamaEmbedClient:
    def __init__(self, spec: ModelSpec, server: dict):
        import httpx
        self.spec = spec
        self.ollama_model = spec.ollama_model
        host = server["host"]
        self.client = httpx.Client(base_url=host, timeout=300)
        r = self.client.post("/api/embed", json={"model": self.ollama_model, "input": "smoke"})
        if r.status_code != 200:
            raise RuntimeError(
                f"Ollama smoke failed (status {r.status_code}): {r.text}. "
                f"Ensure `ollama serve` is running and `ollama pull {self.ollama_model}` has completed."
            )

    def encode(self, texts: list[str], batch_size: int = 8):
        import numpy as np
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            r = self.client.post(
                "/api/embed",
                json={"model": self.ollama_model, "input": batch},
            )
            r.raise_for_status()
            embs = r.json().get("embeddings") or []
            out.extend(embs)
        arr = np.array(out, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class VLLMEmbedClient:
    MAX_INPUT_TOKENS = 8000

    def __init__(self, spec: ModelSpec, server: dict):
        import httpx
        self.spec = spec
        host = server["host"]
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        self.client = httpx.Client(base_url=host, timeout=180, limits=limits)
        r = self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(
                f"vLLM smoke failed (status {r.status_code}). "
                f"Ensure vLLM is running on {host}."
            )
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, trust_remote_code=True)
        except ModuleNotFoundError:
            self.tokenizer = None

    def _truncate(self, text: str) -> str:
        if self.tokenizer is None:
            return text
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.MAX_INPUT_TOKENS:
            return text
        ids = ids[: self.MAX_INPUT_TOKENS]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def encode(self, texts: list[str], batch_size: int = 8):
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        safe_texts = [self._truncate(t) for t in texts]
        slices = [safe_texts[i:i + batch_size] for i in range(0, len(safe_texts), batch_size)]

        def _embed_batch(batch: list[str]) -> list[list[float]]:
            r = self.client.post(
                "/v1/embeddings",
                json={"model": self.spec.hf_repo, "input": batch},
            )
            r.raise_for_status()
            data = r.json().get("data") or []
            return [d["embedding"] for d in data]

        out: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for vecs in ex.map(_embed_batch, slices):
                out.extend(vecs)

        arr = np.array(out, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class LlamaCppEmbedClient:
    def __init__(self, spec: ModelSpec, server: dict):
        import httpx
        self.spec = spec
        self.model_name = spec.llamacpp_model or spec.id
        host = server["host"]
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        self.client = httpx.Client(base_url=host, timeout=300, limits=limits)
        r = self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(
                f"llama.cpp smoke failed (status {r.status_code}). "
                f"Start TurboQuant on {host} with --embedding --pooling last."
            )

    def encode(self, texts: list[str], batch_size: int = 8):
        import numpy as np
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            r = self.client.post(
                "/v1/embeddings",
                json={"model": self.model_name, "input": batch},
            )
            r.raise_for_status()
            data = r.json().get("data") or []
            out.extend([d["embedding"] for d in data])
        arr = np.array(out, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
```

- [ ] **Step 4: Run tests**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/benchmarks/test_embeddings_clients.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add tqbench/benchmarks/embeddings/clients.py tests/benchmarks/test_embeddings_clients.py
git commit -m "feat: embeddings client classes (Gemini, HF, Ollama, vLLM, LlamaCpp)"
```

---

## Task 8: Port runners

**Files:**
- Create: `tqbench/benchmarks/embeddings/runners/__init__.py`
- Create: `tqbench/benchmarks/embeddings/runners/run_all.py`
- Create: `tqbench/benchmarks/embeddings/runners/embed_corpus.py`
- Create: `tqbench/benchmarks/embeddings/runners/eval_quality.py`
- Create: `tqbench/benchmarks/embeddings/runners/eval_speed.py`
- Create: remaining runner files

- [ ] **Step 1: Create runners/__init__.py**

Empty file.

- [ ] **Step 2: Port embed_corpus.py**

Copy `bench_embeddings_turbo/runners/embed_corpus.py` to `tqbench/benchmarks/embeddings/runners/embed_corpus.py`.

Update the imports (lines 14-16 and 28-30) from:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bench.models import get_candidate, load_embedder
from bench.io_lance import bench_lancedb_path
```
to:
```python
from tqbench.benchmarks.embeddings.models import get_candidate, load_client
from tqbench.benchmarks.embeddings.eval.io_lance import bench_lancedb_path
```

Remove the `sys.path.insert` line. Replace `load_embedder` with `load_client` (same interface — `.encode(texts, batch_size)` → ndarray).

Update `SAMPLE_PATH` (line 32) — the relative path to data/ has changed:
```python
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_samples" / "stratified_50k.parquet"
```

- [ ] **Step 3: Port eval_quality.py**

Copy `bench_embeddings_turbo/runners/eval_quality.py` to `tqbench/benchmarks/embeddings/runners/eval_quality.py`.

Update imports from:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bench.io_lance import bench_lancedb_path, read_prod_chunks
from bench.metrics import recall_at_k, mrr_at_k, ndcg_at_k, bootstrap_ci
from bench.models import get_candidate, load_embedder
from bench.snapshot import SNAPSHOT_SEARCH_DUCKDB
from bench.stack import vector_only_retrieve, bm25_retrieve, rrf_fuse
```
to:
```python
from tqbench.benchmarks.embeddings.eval.io_lance import bench_lancedb_path, read_prod_chunks
from tqbench.benchmarks.embeddings.eval.metrics import recall_at_k, mrr_at_k, ndcg_at_k, bootstrap_ci
from tqbench.benchmarks.embeddings.models import get_candidate, load_client
from tqbench.benchmarks.embeddings.eval.snapshot import SNAPSHOT_SEARCH_DUCKDB
from tqbench.benchmarks.embeddings.eval.stack import vector_only_retrieve, bm25_retrieve, rrf_fuse
```

Remove the `sys.path.insert` line. Replace `load_embedder` → `load_client`.

Update `DATA` and `REPORTS` paths:
```python
DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries"
REPORTS = Path(__file__).resolve().parents[1] / "reports" / "raw"
```

- [ ] **Step 4: Port eval_speed.py**

Copy `bench_embeddings_turbo/runners/eval_speed.py` to `tqbench/benchmarks/embeddings/runners/eval_speed.py`.

Update imports:
```python
from tqbench.benchmarks.embeddings.models import get_candidate
```
Remove `sys.path.insert`. Update `REPORTS` and `SAMPLE_PATH`:
```python
REPORTS = Path(__file__).resolve().parents[1] / "reports" / "raw"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_samples" / "stratified_50k.parquet"
```

- [ ] **Step 5: Port remaining runners**

Copy each of these from `bench_embeddings_turbo/runners/` to `tqbench/benchmarks/embeddings/runners/`, applying the same import updates (remove `sys.path.insert`, change `from bench.X` to `from tqbench.benchmarks.embeddings.eval.X` or `from tqbench.benchmarks.embeddings.models`):

- `build_corpus_sample.py`
- `build_layer1.py`
- `build_layer2.py`
- `build_layer2b.py`
- `build_layer2b_via_agent.py`
- `build_adversarial.py`
- `build_dirty.py`
- `extract_session_queries.py`
- `extract_speed_phase1.py`
- `smoke_test_vllm.py`

For each, the pattern is the same:
1. Remove `sys.path.insert(0, ...)` line
2. Change `from bench.<module>` to `from tqbench.benchmarks.embeddings.eval.<module>`
3. Change `from bench.models` to `from tqbench.benchmarks.embeddings.models`
4. Update any `Path(__file__).resolve().parents[N]` paths to account for the new directory depth

- [ ] **Step 6: Port run_all.py**

Copy `bench_embeddings_turbo/runners/run_all.py` to `tqbench/benchmarks/embeddings/runners/run_all.py`.

Update imports:
```python
from tqbench.benchmarks.embeddings.models import load_registry
```

Update `BENCH_ROOT`:
```python
BENCH_ROOT = Path(__file__).resolve().parents[1]
```

Update the runner module paths in `_run()` calls from `runners.X` to `tqbench.benchmarks.embeddings.runners.X`:
```python
_run(sys.executable, "-m", "tqbench.benchmarks.embeddings.runners.build_corpus_sample")
_run(sys.executable, "-m", "tqbench.benchmarks.embeddings.runners.extract_session_queries")
# ... etc for all runner invocations
```

- [ ] **Step 7: Commit**

```bash
git add tqbench/benchmarks/embeddings/runners/
git commit -m "feat: port all embeddings runners with updated imports"
```

---

## Task 9: Port reports and templates

**Files:**
- Create: `tqbench/benchmarks/embeddings/report.py`
- Create: `tqbench/benchmarks/embeddings/report_turboquant_compare.py`
- Create: `tqbench/benchmarks/embeddings/report_turboquant_matrix.py`
- Create: `tqbench/benchmarks/embeddings/report_fp8_cross_query.py`
- Create: `tqbench/benchmarks/embeddings/report_pool_judge.py`
- Create: `tqbench/benchmarks/embeddings/templates/report.html.j2`

- [ ] **Step 1: Copy the Jinja2 template**

Copy `bench_embeddings_turbo/templates/report.html.j2` to `tqbench/benchmarks/embeddings/templates/report.html.j2` unchanged.

- [ ] **Step 2: Port report builders**

Copy each report builder from `bench_embeddings_turbo/runners/` to `tqbench/benchmarks/embeddings/`:

| Source | Destination |
|---|---|
| `runners/build_report.py` | `report.py` |
| `runners/build_turboquant_compare_report.py` | `report_turboquant_compare.py` |
| `runners/build_turboquant_matrix_report.py` | `report_turboquant_matrix.py` |
| `runners/build_fp8_cross_query_report.py` | `report_fp8_cross_query.py` |
| `runners/build_pool_judge_report.py` | `report_pool_judge.py` |

For each, apply the standard import updates:
1. Remove `sys.path.insert(0, ...)` line
2. Change `from bench.<module>` to `from tqbench.benchmarks.embeddings.eval.<module>`
3. Change `from bench.models` to `from tqbench.benchmarks.embeddings.models`
4. Update `Path(__file__).resolve().parents[N]` paths
5. Update template path to `Path(__file__).resolve().parent / "templates" / "report.html.j2"`

- [ ] **Step 3: Create .gitkeep for output dirs**

```bash
mkdir -p tqbench/benchmarks/embeddings/reports/raw
touch tqbench/benchmarks/embeddings/reports/.gitkeep
mkdir -p tqbench/benchmarks/embeddings/indexes
touch tqbench/benchmarks/embeddings/indexes/.gitkeep
mkdir -p tqbench/benchmarks/embeddings/data/eval_queries
touch tqbench/benchmarks/embeddings/data/.gitkeep
mkdir -p tqbench/benchmarks/embeddings/data/chunk_samples
```

- [ ] **Step 4: Commit**

```bash
git add tqbench/benchmarks/embeddings/report*.py tqbench/benchmarks/embeddings/templates/ tqbench/benchmarks/embeddings/reports/.gitkeep tqbench/benchmarks/embeddings/indexes/.gitkeep tqbench/benchmarks/embeddings/data/.gitkeep
git commit -m "feat: port embeddings report builders and templates"
```

---

## Task 10: Integration test — full import chain

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Smoke test that the full import chain works without live servers."""
from __future__ import annotations


def test_import_tqbench():
    import tqbench
    assert tqbench.__version__ == "0.1.0"


def test_import_config():
    from tqbench.config import load_servers, get_server
    servers = load_servers()
    assert "ollama-local" in servers


def test_import_discovery():
    from tqbench.benchmarks import discover_benchmarks
    benchmarks = discover_benchmarks()
    assert "embeddings" in benchmarks


def test_import_embeddings_models():
    from tqbench.benchmarks.embeddings.models import load_registry, get_candidate, baseline_dim
    reg = load_registry()
    assert len(reg) > 0
    spec = get_candidate("gemini-embedding-001")
    assert spec.server == "gemini"
    assert baseline_dim() == 3072


def test_import_embeddings_eval():
    from tqbench.benchmarks.embeddings.eval.metrics import recall_at_k
    from tqbench.benchmarks.embeddings.eval.scoring import decide
    from tqbench.benchmarks.embeddings.eval.stack import rrf_fuse
    from tqbench.benchmarks.embeddings.eval.leakage import is_leaky
    from tqbench.benchmarks.embeddings.eval.perturbations import perturb_all
    from tqbench.benchmarks.embeddings.eval.sampling import stratified_sample
    from tqbench.benchmarks.embeddings.eval.source_weights import weight_for
    from tqbench.benchmarks.embeddings.eval.schemas import ChunkRecord


def test_import_embeddings_clients():
    from tqbench.benchmarks.embeddings.clients import (
        GeminiEmbedClient, HFEmbedClient, OllamaEmbedClient,
        VLLMEmbedClient, LlamaCppEmbedClient, build_client,
    )


def test_cli_list_runs(capsys):
    from tqbench.cli import cmd_list
    from argparse import Namespace
    cmd_list(Namespace())
    captured = capsys.readouterr()
    assert "embeddings" in captured.out
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /mnt/i/dev/LLM/TurboQuant_Benchmark && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke test for full tqbench import chain"
```

---

## Task 11: .gitignore for generated output dirs

**Files:**
- Create or update: `tqbench/benchmarks/embeddings/.gitignore`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Generated output — not committed
indexes/
reports/raw/
reports/*.html
reports/*_summary.json
data/chunk_samples/
data/eval_queries/layer1_pool_judged.jsonl
data/eval_queries/layer2a_perturbed.jsonl
data/eval_queries/layer2b_synthetic.jsonl
data/eval_queries/adversarial_gemini_failures.jsonl
```

- [ ] **Step 2: Commit**

```bash
git add tqbench/benchmarks/embeddings/.gitignore
git commit -m "chore: gitignore for embeddings benchmark generated output"
```

---

## Summary

| Task | What it delivers | Tests |
|---|---|---|
| 1 | Package skeleton + pyproject.toml | — |
| 2 | Shared server config (the only shared code) | 4 tests |
| 3 | Benchmark discovery | 1 test |
| 4 | CLI (list, servers, run) | manual smoke |
| 5 | All eval pure functions (metrics, scoring, sampling, perturbations, leakage, stack, etc.) | 20+ tests |
| 6 | ModelSpec + registry + models.yaml with server refs | 5 tests |
| 7 | Client classes (Gemini, HF, Ollama, vLLM, LlamaCpp) | 3 tests |
| 8 | All runners (embed, eval_quality, eval_speed, build_*, run_all) | — |
| 9 | Report builders + templates | — |
| 10 | Full import chain integration test | 7 tests |
| 11 | .gitignore for generated dirs | — |

Total shared code: `tqbench/config/__init__.py` (~15 lines) + `config/servers.yaml` (~15 lines). Everything else lives inside `tqbench/benchmarks/embeddings/` and can never affect a future sibling benchmark.
