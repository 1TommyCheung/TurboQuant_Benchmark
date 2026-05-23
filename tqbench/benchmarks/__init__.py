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
