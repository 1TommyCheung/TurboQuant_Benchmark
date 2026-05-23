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
        if server["type"] not in ("hf", "gemini_api"):
            assert "host" in server, f"Server '{name}' missing 'host'"
