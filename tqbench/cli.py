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
