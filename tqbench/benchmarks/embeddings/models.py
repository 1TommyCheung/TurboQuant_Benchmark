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
