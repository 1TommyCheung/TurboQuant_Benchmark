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
