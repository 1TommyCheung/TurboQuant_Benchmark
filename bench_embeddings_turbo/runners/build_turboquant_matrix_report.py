from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DATE = date.today().isoformat()

MODELS = [
    {
        "id": "qwen3-embedding-8b-q8-ollama",
        "label": "Ollama qwen3-embedding:8b-q8_0",
        "cache": "Ollama default",
        "server": "ollama",
    },
    {
        "id": "qwen3-embedding-8b-q8-tq-turbo3",
        "label": "TurboQuant turbo3/turbo3",
        "cache": "K turbo3, V turbo3",
        "server": "llama.cpp TurboQuant",
    },
    {
        "id": "qwen3-embedding-8b-q8-tq-turbo4",
        "label": "TurboQuant turbo4/turbo4",
        "cache": "K turbo4, V turbo4",
        "server": "llama.cpp TurboQuant",
    },
    {
        "id": "qwen3-embedding-8b-q8-tq-q8-turbo4",
        "label": "TurboQuant q8_0/turbo4",
        "cache": "K q8_0, V turbo4",
        "server": "llama.cpp TurboQuant",
    },
    {
        "id": "qwen3-embedding-8b-q8-tq-q8-q8",
        "label": "llama.cpp q8_0/q8_0",
        "cache": "K q8_0, V q8_0",
        "server": "llama.cpp TurboQuant build",
    },
]

TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
RATE_RE = re.compile(r"([\d,]+)/([\d,]+)\s+rate=([\d.]+)\s+chunks/s")
MIB_RE = re.compile(r"=\s*([0-9.]+)\s+MiB")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    return datetime.strptime(f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f")


def log_window(path: Path) -> tuple[datetime | None, datetime | None, float | None]:
    if not path.exists():
        return None, None, None
    timestamps = []
    for line in read_text(path).splitlines():
        ts = parse_timestamp(line)
        if ts:
            timestamps.append(ts)
    if not timestamps:
        return None, None, None
    seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    return timestamps[0], timestamps[-1], seconds


def step_log(model_id: str, step: str) -> Path:
    out_path = ROOT / "run_logs" / f"{model_id}.{step}.matrix.out.log"
    err_path = ROOT / "run_logs" / f"{model_id}.{step}.matrix.err.log"
    if err_path.exists() and err_path.stat().st_size:
        return err_path
    return out_path


def parse_embed_rate(path: Path) -> tuple[int | None, float | None]:
    if not path.exists():
        return None, None
    rows = None
    rate = None
    for line in read_text(path).splitlines():
        match = RATE_RE.search(line)
        if match:
            rows = int(match.group(1).replace(",", ""))
            rate = float(match.group(3))
    return rows, rate


def parse_quality(model_id: str) -> dict:
    candidates = sorted((ROOT / "reports" / "raw").glob(f"*_ {model_id}_quality.json"))
    if not candidates:
        candidates = sorted((ROOT / "reports" / "raw").glob(f"*_{model_id}_quality.json"))
    if not candidates:
        return {}
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def parse_cross_quality(query_model_id: str, index_model_id: str) -> dict:
    candidates = sorted((ROOT / "reports" / "raw").glob(f"*_{query_model_id}_on_{index_model_id}_quality.json"))
    if not candidates:
        return {}
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def parse_vram_csv(path: Path) -> int | None:
    if not path.exists():
        return None
    values = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(int(row["memory_used_mib"].strip()))
            except Exception:
                continue
    return max(values) if values else None


def parse_server_vram(model_id: str) -> dict[str, float | None]:
    path = ROOT / "run_logs" / f"{model_id}.server.matrix.err.log"
    out = {
        "gpu_model_mib": None,
        "kv_mib": None,
        "compute_mib": None,
        "host_compute_mib": None,
    }
    if not path.exists():
        return out
    for line in read_text(path).splitlines():
        match = MIB_RE.search(line)
        if not match:
            continue
        val = float(match.group(1))
        if "CUDA0 model buffer size" in line:
            out["gpu_model_mib"] = val
        elif "CUDA0 KV buffer size" in line:
            out["kv_mib"] = val
        elif "CUDA0 compute buffer size" in line:
            out["compute_mib"] = val
        elif "CUDA_Host compute buffer size" in line:
            out["host_compute_mib"] = val
    return out


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def num(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value:,.2f}{suffix}"


def minutes(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return f"{seconds / 60:.1f} min"


def row(model: dict) -> dict:
    model_id = model["id"]
    embed_log = step_log(model_id, "embed")
    eval_log = step_log(model_id, "eval")
    _, _, embed_seconds = log_window(embed_log)
    _, _, eval_seconds = log_window(eval_log)
    rows, rate = parse_embed_rate(embed_log)
    quality = parse_quality(model_id)
    metrics = (quality.get("aggregate") or {}).get("overall_mean") or {}
    n_queries = quality.get("n_queries")
    query_rate = (n_queries / eval_seconds) if n_queries and eval_seconds else None
    peak_embed = parse_vram_csv(ROOT / "run_logs" / f"{model_id}.embed.matrix.vram.csv")
    peak_eval = parse_vram_csv(ROOT / "run_logs" / f"{model_id}.eval.matrix.vram.csv")
    server_vram = parse_server_vram(model_id)
    return {
        **model,
        "rows": rows,
        "embed_rate": rate,
        "embed_seconds": embed_seconds,
        "eval_seconds": eval_seconds,
        "n_queries": n_queries,
        "query_rate": query_rate,
        "metrics": metrics,
        "peak_embed_mib": peak_embed,
        "peak_eval_mib": peak_eval,
        "peak_mib": max([v for v in (peak_embed, peak_eval) if v is not None], default=None),
        **server_vram,
    }


def html(rows: list[dict]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table_rows = "\n".join(
        "<tr>"
        f"<th>{escape(r['label'])}</th>"
        f"<td>{escape(r['cache'])}</td>"
        f"<td>{num(r['embed_rate'], ' chunks/s')}</td>"
        f"<td>{minutes(r['embed_seconds'])}</td>"
        f"<td>{num(r['query_rate'], ' q/s')}</td>"
        f"<td>{minutes(r['eval_seconds'])}</td>"
        f"<td>{pct(r['metrics'].get('vec_recall_10'))}</td>"
        f"<td>{pct(r['metrics'].get('vec_mrr_10'))}</td>"
        f"<td>{pct(r['metrics'].get('e2e_recall_10'))}</td>"
        f"<td>{pct(r['metrics'].get('e2e_mrr_10'))}</td>"
        f"<td>{num(r['peak_mib'], ' MiB')}</td>"
        f"<td>{num(r['kv_mib'], ' MiB')}</td>"
        "</tr>"
        for r in rows
    )
    detail_rows = "\n".join(
        "<tr>"
        f"<th>{escape(r['label'])}</th>"
        f"<td>{num(r['gpu_model_mib'], ' MiB')}</td>"
        f"<td>{num(r['kv_mib'], ' MiB')}</td>"
        f"<td>{num(r['compute_mib'], ' MiB')}</td>"
        f"<td>{num(r['peak_embed_mib'], ' MiB')}</td>"
        f"<td>{num(r['peak_eval_mib'], ' MiB')}</td>"
        "</tr>"
        for r in rows
    )
    cross_index_models = [
        ("qwen3-embedding-8b-q8-ollama", "Ollama qwen3-embedding:8b-q8_0"),
    ]
    cross_rows = []
    for r in rows:
        for index_id, index_label in cross_index_models:
            if r["id"] == index_id:
                continue
            q = parse_cross_quality(r["id"], index_id)
            metrics = (q.get("aggregate") or {}).get("overall_mean") or {}
            if not metrics:
                continue
            cross_rows.append(
                "<tr>"
                f"<th>{escape(r['label'])}</th>"
                f"<td>{escape(index_label)}</td>"
                f"<td>{pct(metrics.get('vec_recall_10'))}</td>"
                f"<td>{pct(metrics.get('vec_mrr_10'))}</td>"
                f"<td>{pct(metrics.get('e2e_recall_10'))}</td>"
                f"<td>{pct(metrics.get('e2e_mrr_10'))}</td>"
                "</tr>"
            )
    cross_table_rows = "\n".join(cross_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TurboQuant Cache Matrix</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; color: #17202a; background: #fff; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 34px 26px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 30px 0 10px; font-size: 22px; letter-spacing: 0; }}
    p {{ color: #5a6573; margin: 0 0 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0 24px; }}
    th, td {{ border-bottom: 1px solid #d8dee7; padding: 9px 10px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{ background: #eef3f7; color: #27313d; font-weight: 650; }}
    code {{ background: #eef3f7; padding: 2px 5px; border-radius: 4px; }}
    .note {{ border-left: 4px solid #0f766e; background: #f0fdfa; padding: 12px 14px; color: #24413e; }}
  </style>
</head>
<body>
<main>
  <h1>TurboQuant Cache Matrix</h1>
  <p>Generated {escape(generated)} from <code>{escape(str(ROOT))}</code>.</p>
  <div class="note">This report compares Ollama against four llama.cpp/TurboQuant cache modes on the same Qwen3-Embedding-8B Q8_0 GGUF, using the configured corpus slice and evaluation query set. Peak VRAM is sampled with <code>nvidia-smi</code>; llama.cpp per-buffer values come from server startup logs.</div>

  <h2>Speed, Accuracy, VRAM</h2>
  <table>
    <thead>
      <tr><th>Model</th><th>Cache</th><th>Chunk speed</th><th>Chunk wall time</th><th>Query speed</th><th>Query wall time</th><th>Vec recall@10</th><th>Vec MRR@10</th><th>E2E recall@10</th><th>E2E MRR@10</th><th>Peak VRAM</th><th>KV cache</th></tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>

  <h2>VRAM Detail</h2>
  <table>
    <thead>
      <tr><th>Model</th><th>GPU model buffer</th><th>KV cache</th><th>CUDA compute buffer</th><th>Peak embed VRAM</th><th>Peak eval VRAM</th></tr>
    </thead>
    <tbody>{detail_rows}</tbody>
  </table>

  <h2>Cross-Geometry Query Test</h2>
  <p>These rows embed the corpus chunks once with Ollama at the shared 32K context, then embed only the query with the listed model. This directly tests whether a TurboQuant query vector can safely search an Ollama-built LanceDB index without changing the context-window assumption.</p>
  <table>
    <thead>
      <tr><th>Query model</th><th>Chunk/index model</th><th>Vec recall@10</th><th>Vec MRR@10</th><th>E2E recall@10</th><th>E2E MRR@10</th></tr>
    </thead>
    <tbody>{cross_table_rows}</tbody>
  </table>
</main>
</body>
</html>
"""


def main() -> None:
    rows = [row(m) for m in MODELS]
    out = ROOT / "reports" / f"{REPORT_DATE}_qwen3_q8_turboquant_matrix.html"
    out.write_text(html(rows), encoding="utf-8")
    summary = ROOT / "reports" / f"{REPORT_DATE}_qwen3_q8_turboquant_matrix_summary.json"
    summary.write_text(json.dumps({"rows": rows, "report": str(out)}, indent=2), encoding="utf-8")
    print(out)
    print(summary)


if __name__ == "__main__":
    main()
