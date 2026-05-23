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
