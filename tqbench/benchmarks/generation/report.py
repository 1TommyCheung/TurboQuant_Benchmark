"""HTML comparison report builder for the generation benchmark."""
from __future__ import annotations
import datetime as dt
import json
import logging
from pathlib import Path

import jinja2

from tqbench.benchmarks.generation.models import load_registry

BENCH_ROOT = Path(__file__).resolve().parent
REPORTS = BENCH_ROOT / "reports"
RAW = REPORTS / "raw"
TEMPLATES = BENCH_ROOT / "templates"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_speed_data() -> dict[str, dict]:
    out = {}
    for f in RAW.glob("*_speed.json"):
        data = json.loads(f.read_text())
        out[data["model_id"]] = data
    return out


def _load_ttft_longctx_data() -> dict[str, dict]:
    out = {}
    for f in RAW.glob("*_ttft_longctx.json"):
        data = json.loads(f.read_text())
        out[data["model_id"]] = data
    return out


def _load_quality_data() -> dict[str, dict]:
    out = {}
    for d in RAW.iterdir():
        if d.is_dir() and "_quality" in d.name:
            for f in d.rglob("results.json"):
                data = json.loads(f.read_text())
                out[d.name] = data
    return out


def main() -> None:
    registry = load_registry()
    date = dt.date.today().isoformat()

    configs = [
        {"id": c.id, "server": c.server,
         "spec_decode": c.spec_decode, "spec_prefill": c.spec_prefill}
        for c in registry
    ]

    speed_data = _load_speed_data()
    ttft_data = _load_ttft_longctx_data()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES)),
        autoescape=False,
    )
    tmpl = env.get_template("report.html.j2")

    html = tmpl.render(
        run_date=date,
        n_configs=len(registry),
        configs=configs,
        quality_summary=None,
        throughput_chart=None,
        ttft_chart=None,
        itl_chart=None,
        longctx_chart=None,
        vram_chart=None,
        spec_decode_chart=None,
        latency_rows=None,
    )

    out_path = REPORTS / f"{date}_generation_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
