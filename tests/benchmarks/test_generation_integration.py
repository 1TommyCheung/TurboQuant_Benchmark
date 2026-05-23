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
