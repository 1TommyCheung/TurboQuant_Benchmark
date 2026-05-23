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
