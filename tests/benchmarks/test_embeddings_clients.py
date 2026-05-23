from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import MagicMock
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
