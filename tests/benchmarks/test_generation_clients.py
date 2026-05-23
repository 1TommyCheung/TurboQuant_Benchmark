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
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}',
        'data: [DONE]',
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.raise_for_status = MagicMock()

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


def test_build_client_raises_when_server_down():
    spec = _spec()
    with pytest.raises(Exception):
        build_client(spec)
