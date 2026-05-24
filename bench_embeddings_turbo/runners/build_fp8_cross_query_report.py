from __future__ import annotations

import json
import re
from datetime import date, datetime
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
REPORT_DATE = date.today().isoformat()
INDEX_ID = "qwen3-embedding-8b-fp8-vllm"
DOCKER_ID = "qwen3-embedding-8b-fp8-vllm-docker-8k"
ORIGINAL_QUALITY = Path(
    r"I:\dev\Legal\case_kb\bench_embeddings\reports\raw\2026-05-16_qwen3-embedding-8b-fp8-vllm_quality.json"
)
ORIGINAL_REPORT = Path(r"I:\dev\Legal\case_kb\bench_embeddings\reports\2026-05-17_pool_and_judge_report.html")
FP8_INDEX_PATH = Path(r"I:\dev\Legal\case_kb\bench_embeddings\indexes\qwen3-embedding-8b-fp8-vllm.lance")

QUERY_MODELS = [
    (INDEX_ID, "Original FP8/vLLM artifact", "vLLM", "compressed-tensors FP8", None, None),
    (DOCKER_ID, "Docker/WSL2 FP8 vLLM 8K rerun", "vLLM Docker on WSL2", "compressed-tensors FP8", None, None),
    ("qwen3-embedding-8b-q8-tq-turbo3", "TurboQuant Q8 turbo3/turbo3", "llama.cpp TurboQuant", "Q8_0 GGUF", "turbo3", "turbo3"),
    ("qwen3-embedding-8b-q8-tq-turbo4", "TurboQuant Q8 turbo4/turbo4", "llama.cpp TurboQuant", "Q8_0 GGUF", "turbo4", "turbo4"),
    ("qwen3-embedding-8b-q8-tq-q8-turbo4", "TurboQuant Q8 q8_0/turbo4", "llama.cpp TurboQuant", "Q8_0 GGUF", "q8_0", "turbo4"),
    ("qwen3-embedding-8b-q8-tq-q8-q8", "llama.cpp Q8 q8_0/q8_0", "llama.cpp TurboQuant build", "Q8_0 GGUF", "q8_0", "q8_0"),
]

RUN_PARAMETERS = {
    "TurboQuant binary": str(REPO_ROOT / "turboquant-plus-tqp-v0.1.1-windows-x64-cuda12.4" / "llama-server.exe"),
    "TurboQuant model": str(REPO_ROOT / "models" / "Qwen3-Embedding-8B-Q8_0.gguf"),
    "TurboQuant model size": "8,047,105,824 bytes",
    "TurboQuant server flags": "--embedding --pooling last -ngl all -c 8192 -b 8192 -ub 8192 -fa on -np 1 --metrics --host 0.0.0.0 --port 8080",
    "TurboQuant per-scenario flags": "--cache-type-k / --cache-type-v varied by row",
    "Docker/WSL2 vLLM image": "vllm/vllm-openai:gemma4-cu130",
    "Docker/WSL2 vLLM model": "maywell/Qwen3-Embedding-8B-FP8-Dynamic",
    "Docker/WSL2 vLLM server flags": "vllm serve maywell/Qwen3-Embedding-8B-FP8-Dynamic --runner pooling --convert embed --max-model-len 8192 --served-model-name maywell/Qwen3-Embedding-8B-FP8-Dynamic --host 0.0.0.0 --port 8000 --trust-remote-code --gpu-memory-utilization 0.90",
    "Original FP8/vLLM model": "maywell/Qwen3-Embedding-8B-FP8-Dynamic",
    "Original FP8/vLLM quantization": "compressed_tensors / fp8_dynamic",
    "Original FP8/vLLM dimension": "4096",
    "Original FP8/vLLM context in registry": "32768 tokens",
    "Original FP8/vLLM speed artifact": "not found; original report notes Phase 2 speed was not run",
    "Index under test": str(FP8_INDEX_PATH),
    "Benchmark mode": "8K query-only compatibility test; no corpus re-embedding",
}


def load_quality(query_id: str) -> dict:
    if query_id == INDEX_ID and ORIGINAL_QUALITY.exists():
        return json.loads(ORIGINAL_QUALITY.read_text(encoding="utf-8"))
    matches = sorted((ROOT / "reports" / "raw").glob(f"*_{query_id}_on_{INDEX_ID}_quality.json"))
    if not matches:
        return {}
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def parse_query_timing(query_id: str) -> tuple[float | None, float | None]:
    log_path = ROOT / "run_logs" / f"{query_id}.on-fp8-vllm-8k.eval.err.log"
    if not log_path.exists():
        return None, None
    start = None
    end = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if start is None and "HTTP Request" in line:
            match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if match:
                start = datetime.fromisoformat(match.group(1))
        if "Wrote" in line:
            match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if match:
                end = datetime.fromisoformat(match.group(1))
    if not start or not end:
        return None, None
    seconds = (end - start).total_seconds()
    return (seconds, 893 / seconds) if seconds > 0 else (None, None)


def parse_server_buffers(query_id: str) -> dict[str, float | None]:
    log_path = ROOT / "run_logs" / f"{query_id}.fp8-cross-8k.server.err.log"
    values = {"gpu_model_mib": None, "kv_mib": None, "compute_mib": None, "host_compute_mib": None}
    if not log_path.exists():
        return values
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "gpu_model_mib": r"CUDA0 model buffer size\s*=\s*([0-9.]+)\s*MiB",
        "kv_mib": r"CUDA0 KV buffer size\s*=\s*([0-9.]+)\s*MiB",
        "compute_mib": r"CUDA0 compute buffer size\s*=\s*([0-9.]+)\s*MiB",
        "host_compute_mib": r"CUDA_Host compute buffer size\s*=\s*([0-9.]+)\s*MiB",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = float(match.group(1))
    return values


def parse_peak_vram(query_id: str) -> int | None:
    candidates = [
        ROOT / "run_logs" / f"{query_id}.on-fp8-vllm-8k.eval.vram.csv",
        ROOT / "run_logs" / f"{query_id}.eval.matrix.vram.csv",
    ]
    peak = None
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                used = int(float(parts[1]))
            except ValueError:
                continue
            peak = used if peak is None else max(peak, used)
    return peak


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_rows() -> list[dict]:
    rows = []
    for model_id, label, stack, precision, cache_k, cache_v in QUERY_MODELS:
        quality = load_quality(model_id)
        metrics = (quality.get("aggregate") or {}).get("overall_mean") or {}
        seconds, qps = parse_query_timing(model_id)
        rows.append(
            {
                "query_model_id": model_id,
                "query_model": label,
                "inference_stack": stack,
                "precision": precision,
                "cache_k": cache_k,
                "cache_v": cache_v,
                "index_model_id": INDEX_ID,
                "n_queries": quality.get("n_queries"),
                "query_seconds": seconds,
                "query_rate": qps,
                **metrics,
                **parse_server_buffers(model_id),
                "peak_vram_mib": parse_peak_vram(model_id),
            }
        )
    return rows


def row_cells(r: dict) -> list[str]:
    return [
        r["query_model"],
        r["inference_stack"],
        r["precision"],
        str(r.get("cache_k") or "n/a"),
        str(r.get("cache_v") or "n/a"),
        str(r.get("n_queries") or "n/a"),
        num(r.get("query_seconds"), 1),
        num(r.get("query_rate"), 3),
        pct(r.get("vec_recall_10")),
        pct(r.get("vec_mrr_10")),
        pct(r.get("e2e_recall_10")),
        pct(r.get("e2e_mrr_10")),
        num(r.get("peak_vram_mib"), 0),
        num(r.get("kv_mib"), 2),
        num(r.get("gpu_model_mib"), 2),
        num(r.get("compute_mib"), 2),
    ]


def render_html(rows: list[dict], generated: str) -> str:
    metric_rows = "\n".join(
        "<tr>" + "".join(
            f"<th>{escape(c)}</th>" if i == 0 else f"<td>{escape(c)}</td>"
            for i, c in enumerate(row_cells(r))
        ) + "</tr>"
        for r in rows
    )
    params = "\n".join(f"<tr><th>{escape(k)}</th><td><code>{escape(v)}</code></td></tr>" for k, v in RUN_PARAMETERS.items())
    measured = [r for r in rows if r.get("query_rate") is not None]
    max_qps = max((r["query_rate"] for r in measured), default=1.0)
    max_vram = max((r["peak_vram_mib"] or 0 for r in rows), default=1)
    max_accuracy = max((r.get("e2e_mrr_10") or 0 for r in rows), default=1)

    def short_label(label: str) -> str:
        return (
            label.replace("Docker/WSL2 FP8 vLLM 8K rerun", "vLLM FP8 Docker")
            .replace("TurboQuant Q8 ", "TQ ")
            .replace("llama.cpp Q8 ", "llama.cpp ")
            .replace("Original FP8/vLLM artifact", "Original FP8")
        )

    def bar_row(r: dict, key: str, max_value: float, suffix: str, digits: int = 3) -> str:
        value = r.get(key)
        width = 0 if value is None or max_value <= 0 else max(2, (value / max_value) * 100)
        value_text = "n/a" if value is None else f"{value:.{digits}f}{suffix}"
        return (
            "<div class=\"bar-row\">"
            f"<div class=\"bar-label\">{escape(short_label(r['query_model']))}</div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar-fill\" style=\"width:{width:.2f}%\"></div>"
            "</div>"
            f"<div class=\"bar-value\">{escape(value_text)}</div>"
            "</div>"
        )

    speed_chart = "\n".join(bar_row(r, "query_rate", max_qps, " q/s") for r in rows if r["query_model_id"] != INDEX_ID)
    vram_chart = "\n".join(bar_row(r, "peak_vram_mib", max_vram, " MiB", 0) for r in rows if r.get("peak_vram_mib") is not None)
    accuracy_chart = "\n".join(bar_row(r, "e2e_mrr_10", max_accuracy, "", 4) for r in rows)
    recall_chart = "\n".join(bar_row(r, "e2e_recall_10", max((x.get("e2e_recall_10") or 0 for x in rows), default=1), "", 4) for r in rows)

    docker_row = next((r for r in rows if r["query_model_id"] == DOCKER_ID), {})
    best_speed = max((r for r in rows if r.get("query_rate") is not None), key=lambda r: r["query_rate"], default={})
    best_mrr = max((r for r in rows if r.get("e2e_mrr_10") is not None), key=lambda r: r["e2e_mrr_10"], default={})
    best_vram = min((r for r in rows if r.get("peak_vram_mib") is not None), key=lambda r: r["peak_vram_mib"], default={})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3 8B FP8 Index / TurboQuant Query Comparison</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; color: #17202a; background: #fff; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 34px 26px 54px; }}
    h1 {{ margin: 0 0 8px; font-size: 31px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 10px; font-size: 20px; }}
    p {{ color: #566371; margin: 0 0 14px; line-height: 1.45; }}
    .hero {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 22px; align-items: end; margin-bottom: 18px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0 8px; }}
    .kpi {{ border: 1px solid #d8dee7; border-radius: 8px; padding: 14px 16px; background: #fafbfd; }}
    .kpi .label {{ color: #5e6b78; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi .value {{ color: #111827; font-size: 24px; font-weight: 720; margin-top: 7px; }}
    .kpi .sub {{ color: #5e6b78; font-size: 12px; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }}
    .panel {{ border: 1px solid #d8dee7; border-radius: 8px; padding: 16px; background: #fff; }}
    .panel h3 {{ margin: 0 0 12px; font-size: 16px; }}
    .bar-row {{ display: grid; grid-template-columns: 160px 1fr 94px; gap: 10px; align-items: center; margin: 9px 0; }}
    .bar-label {{ font-size: 12px; color: #27313d; overflow-wrap: anywhere; }}
    .bar-track {{ height: 18px; background: #eef3f7; border-radius: 5px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, #0f766e, #38bdf8); border-radius: 5px; }}
    .bar-value {{ font-variant-numeric: tabular-nums; font-size: 12px; color: #27313d; text-align: right; }}
    .panel:nth-child(2) .bar-fill {{ background: linear-gradient(90deg, #b45309, #f59e0b); }}
    .panel:nth-child(3) .bar-fill {{ background: linear-gradient(90deg, #4338ca, #8b5cf6); }}
    .panel:nth-child(4) .bar-fill {{ background: linear-gradient(90deg, #be123c, #fb7185); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid #d8dee7; padding: 9px 10px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{ background: #eef3f7; color: #27313d; font-weight: 650; white-space: nowrap; }}
    .note {{ border-left: 4px solid #0f766e; background: #f0fdfa; padding: 12px 14px; color: #24413e; }}
    .warn {{ border-left-color: #b45309; background: #fffbeb; color: #4f3820; }}
    code {{ background: #eef3f7; padding: 2px 5px; border-radius: 4px; white-space: pre-wrap; }}
    .params th {{ width: 260px; }}
    .params td {{ text-align: left; }}
  </style>
</head>
<body>
<main>
  <div class="hero">
    <div>
      <h1>Qwen3 8B FP8 Index / TurboQuant Query Comparison</h1>
      <p>Generated {escape(generated)} from <code>{escape(str(ROOT))}</code>.</p>
    </div>
  </div>
  <div class="note">This lines up the original Qwen3-Embedding-8B FP8/vLLM quality artifact, the Docker/WSL2 vLLM 8K rerun, and the TurboQuant query-only compatibility run. The rerun rows reuse the existing FP8/vLLM LanceDB vectors and only swap the query embedding stack.</div>
  <div class="note warn">The original FP8/vLLM row is accuracy-only. Its old ~9 GB VRAM note appears to be a model/runtime estimate and should not be read as active total VRAM including comparable KV/cache allocation.</div>
  <section class="kpis">
    <div class="kpi"><div class="label">Fastest measured query</div><div class="value">{escape(short_label(best_speed.get('query_model', 'n/a')))}</div><div class="sub">{num(best_speed.get('query_rate'), 3)} q/s</div></div>
    <div class="kpi"><div class="label">Best E2E MRR@10</div><div class="value">{escape(short_label(best_mrr.get('query_model', 'n/a')))}</div><div class="sub">{pct(best_mrr.get('e2e_mrr_10'))}</div></div>
    <div class="kpi"><div class="label">Lowest measured peak VRAM</div><div class="value">{escape(short_label(best_vram.get('query_model', 'n/a')))}</div><div class="sub">{num(best_vram.get('peak_vram_mib'), 0)} MiB</div></div>
  </section>

  <h2>Graph Comparison</h2>
  <section class="grid">
    <div class="panel"><h3>Query Speed</h3>{speed_chart}</div>
    <div class="panel"><h3>Peak VRAM</h3>{vram_chart}</div>
    <div class="panel"><h3>E2E MRR@10</h3>{accuracy_chart}</div>
    <div class="panel"><h3>E2E Recall@10</h3>{recall_chart}</div>
  </section>

  <h2>Results</h2>
  <table>
    <thead><tr><th>Model</th><th>Stack</th><th>Precision</th><th>K cache</th><th>V cache</th><th>Queries</th><th>Seconds</th><th>q/s</th><th>Vec R@10</th><th>Vec MRR@10</th><th>E2E R@10</th><th>E2E MRR@10</th><th>Peak VRAM MiB</th><th>KV MiB</th><th>GPU model MiB</th><th>Compute MiB</th></tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>
  <h2>Run Parameters</h2>
  <table class="params"><tbody>{params}</tbody></table>
  <h2>Interpretation</h2>
  <p>Best measured query stack is TurboQuant <code>q8_0/turbo4</code>: it ran at <code>1.368 q/s</code>, ahead of the Docker/WSL2 FP8 vLLM rerun at <code>1.042 q/s</code>, while also posting the strongest E2E MRR@10 in this 8K compatibility run.</p>
  <p>The Docker/WSL2 FP8 vLLM row is now timed, so the graph compares speed, accuracy, and sampled active VRAM across the measured stacks. TurboQuant can query the existing FP8/vLLM LanceDB vectors without re-embedding the corpus, but lower-precision query embeddings may shift ranking quality depending on cache setting.</p>
  <p>At 8K, TurboQuant server logs show KV cache ranging from roughly <code>225 MiB</code> for turbo3/turbo3 to <code>612 MiB</code> for q8_0/q8_0. Where available, peak VRAM is sampled by <code>nvidia-smi</code> during the eval pass.</p>
</main>
</body>
</html>
"""


def render_markdown(rows: list[dict], generated: str, html_path: Path) -> str:
    lines = [
        "# Qwen3 8B FP8 Index / TurboQuant Query Comparison",
        "",
        f"Generated: {generated}",
        f"HTML report: `{html_path}`",
        "",
        "This report compares the original Qwen3-Embedding-8B FP8/vLLM quality artifact, the Docker/WSL2 vLLM 8K rerun, and the TurboQuant query-only compatibility run.",
        "",
        "Important caveat: the original FP8/vLLM row is accuracy-only. Its old `~9 GB` VRAM note appears to be a model/runtime estimate, not active total VRAM including comparable KV/cache allocation.",
        "",
        "## Results",
        "",
        "| Model | Stack | Precision | K cache | V cache | Queries | Seconds | q/s | Vec R@10 | Vec MRR@10 | E2E R@10 | E2E MRR@10 | Peak VRAM MiB | KV MiB | GPU model MiB | Compute MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(row_cells(r)) + " |")
    lines.extend(["", "## Run Parameters", ""])
    for key, value in RUN_PARAMETERS.items():
        lines.append(f"- **{key}:** `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Best measured query stack is TurboQuant `q8_0/turbo4`: it ran at `1.368 q/s`, ahead of the Docker/WSL2 FP8 vLLM rerun at `1.042 q/s`, while also posting the strongest E2E MRR@10 in this 8K compatibility run.",
            "",
            "The Docker/WSL2 FP8 vLLM row is now timed, so the graph compares speed, accuracy, and sampled active VRAM across the measured stacks. TurboQuant can query the existing FP8/vLLM LanceDB vectors without re-embedding the corpus, but lower-precision query embeddings may shift ranking quality depending on cache setting.",
            "",
            "At 8K, TurboQuant server logs show KV cache ranging from roughly `225 MiB` for turbo3/turbo3 to `612 MiB` for q8_0/q8_0. Where available, peak VRAM is sampled by `nvidia-smi` during the eval pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = build_rows()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_html = ROOT / "reports" / f"{REPORT_DATE}_fp8_vllm_index_turboquant_query_8k.html"
    out_md = ROOT / "reports" / f"{REPORT_DATE}_fp8_vllm_index_turboquant_query_8k.md"
    summary = ROOT / "reports" / f"{REPORT_DATE}_fp8_vllm_index_turboquant_query_8k_summary.json"
    out_html.write_text(render_html(rows, generated), encoding="utf-8")
    out_md.write_text(render_markdown(rows, generated, out_html), encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "rows": rows,
                "parameters": RUN_PARAMETERS,
                "report": str(out_html),
                "markdown": str(out_md),
                "original_report": str(ORIGINAL_REPORT),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_html)
    print(out_md)
    print(summary)


if __name__ == "__main__":
    main()
