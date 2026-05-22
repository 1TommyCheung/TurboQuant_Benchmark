from __future__ import annotations

import json
import re
from datetime import datetime
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT.parent
REPORT_DATE = "2026-05-22"

MODELS = {
    "ollama": {
        "label": "Ollama qwen3-embedding:8b-q8_0",
        "model_id": "qwen3-embedding-8b-q8-ollama",
        "quality": ROOT / "reports" / "raw" / f"{REPORT_DATE}_qwen3-embedding-8b-q8-ollama_quality.json",
        "embed_log": ROOT / "run_logs" / "qwen3-embedding-8b-q8-ollama.embed.5k.out.log",
        "eval_log": ROOT / "run_logs" / "qwen3-embedding-8b-q8-ollama.eval.5k.out.log",
        "index": ROOT / "indexes" / "qwen3-embedding-8b-q8-ollama.lance",
        "model_size": "8.0 GB via ollama list",
    },
    "turbo": {
        "label": "TurboQuant llama.cpp Qwen3-Embedding-8B Q8_0",
        "model_id": "qwen3-embedding-8b-q8-turbo",
        "quality": ROOT / "reports" / "raw" / f"{REPORT_DATE}_qwen3-embedding-8b-q8-turbo_quality.json",
        "embed_log": ROOT / "run_logs" / "qwen3-embedding-8b-q8-turbo.embed.5k.out.log",
        "eval_log": ROOT / "run_logs" / "qwen3-embedding-8b-q8-turbo.eval.5k.out.log",
        "index": ROOT / "indexes" / "qwen3-embedding-8b-q8-turbo.lance",
        "model_path": BENCH_ROOT / "models" / "Qwen3-Embedding-8B-Q8_0.gguf",
    },
}

TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
RATE_RE = re.compile(r"([\d,]+)/([\d,]+)\s+rate=([\d.]+)\s+chunks/s")


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    return datetime.strptime(f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f")


def read_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def log_window(path: Path) -> tuple[datetime, datetime]:
    timestamps = []
    for line in read_log(path).splitlines():
        ts = parse_timestamp(line)
        if ts:
            timestamps.append(ts)
    if not timestamps:
        raise ValueError(f"No timestamps found in {path}")
    return timestamps[0], timestamps[-1]


def embed_rate(path: Path) -> tuple[int, float]:
    rows = 0
    rate = 0.0
    for line in read_log(path).splitlines():
        match = RATE_RE.search(line)
        if match:
            rows = int(match.group(1).replace(",", ""))
            rate = float(match.group(3))
    if not rows or not rate:
        raise ValueError(f"No final rate found in {path}")
    return rows, rate


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def gb(size: int) -> str:
    return f"{size / 1_000_000_000:.2f} GB"


def gib(size: int) -> str:
    return f"{size / (1024 ** 3):.2f} GiB"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta(new: float, base: float) -> str:
    return f"{(new - base) * 100:+.2f} pp"


def seconds_label(seconds: float) -> str:
    minutes = seconds / 60
    return f"{minutes:.1f} min"


def load_result(key: str) -> dict:
    spec = MODELS[key]
    quality = json.loads(spec["quality"].read_text(encoding="utf-8"))
    embed_start, embed_end = log_window(spec["embed_log"])
    eval_start, eval_end = log_window(spec["eval_log"])
    rows, rate = embed_rate(spec["embed_log"])
    model_size = spec.get("model_size")
    if "model_path" in spec:
        size = dir_size(spec["model_path"])
        model_size = f"{gb(size)} / {gib(size)}"
    index_size = dir_size(spec["index"])
    return {
        "label": spec["label"],
        "model_id": spec["model_id"],
        "rows": rows,
        "embed_rate": rate,
        "embed_seconds": (embed_end - embed_start).total_seconds(),
        "eval_seconds": (eval_end - eval_start).total_seconds(),
        "n_queries": quality["n_queries"],
        "metrics": quality["aggregate"]["overall_mean"],
        "by_source_type": quality["aggregate"].get("by_source_type", {}),
        "model_size": model_size,
        "index_size_bytes": index_size,
        "index_size": f"{gb(index_size)} / {gib(index_size)}",
    }


def metric_row(name: str, key: str, ollama: dict, turbo: dict) -> str:
    ov = ollama["metrics"][key]
    tv = turbo["metrics"][key]
    winner = "TurboQuant" if tv > ov else "Ollama" if ov > tv else "Tie"
    return (
        "<tr>"
        f"<th>{escape(name)}</th>"
        f"<td>{pct(ov)}</td>"
        f"<td>{pct(tv)}</td>"
        f"<td>{delta(tv, ov)}</td>"
        f"<td>{winner}</td>"
        "</tr>"
    )


def source_rows(ollama: dict, turbo: dict) -> str:
    sources = sorted(set(ollama["by_source_type"]) | set(turbo["by_source_type"]))
    rows = []
    for source in sources:
        ov = ollama["by_source_type"].get(source, 0.0)
        tv = turbo["by_source_type"].get(source, 0.0)
        rows.append(
            "<tr>"
            f"<th>{escape(source)}</th>"
            f"<td>{pct(ov)}</td>"
            f"<td>{pct(tv)}</td>"
            f"<td>{delta(tv, ov)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(ollama: dict, turbo: dict) -> str:
    speed_delta = (turbo["embed_rate"] / ollama["embed_rate"] - 1.0) * 100
    eval_delta = (turbo["eval_seconds"] / ollama["eval_seconds"] - 1.0) * 100
    fp16_kv_mib = 32768 * 36 * (1024 + 1024) * 2 / (1024**2)
    turbo_kv_mib = 900.0
    kv_saved_mib = fp16_kv_mib - turbo_kv_mib
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TurboQuant vs Ollama Qwen3 Embedding 8B Q8 Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5a6573;
      --line: #d8dee7;
      --panel: #f7f9fc;
      --accent: #0f766e;
      --warn: #9a3412;
    }}
    body {{
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.45;
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 36px 28px 48px;
    }}
    h1, h2 {{
      line-height: 1.15;
      margin: 0 0 14px;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 22px; margin-top: 34px; }}
    p {{ margin: 0 0 14px; color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 24px 0;
    }}
    .tile {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: var(--panel);
    }}
    .tile strong {{
      display: block;
      font-size: 24px;
      margin-top: 6px;
      color: var(--ink);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 24px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: right;
      vertical-align: top;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{
      background: #eef3f7;
      color: #27313d;
      font-weight: 650;
    }}
    .note {{
      border-left: 4px solid var(--warn);
      background: #fff7ed;
      padding: 12px 14px;
      color: #4b2e1f;
      margin: 18px 0 24px;
    }}
    .ok {{ color: var(--accent); font-weight: 650; }}
    code {{
      background: #eef3f7;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 24px 16px; }}
      .summary {{ grid-template-columns: 1fr; }}
      table {{ font-size: 13px; }}
      th, td {{ padding: 8px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>TurboQuant vs Ollama: Qwen3 Embedding 8B Q8</h1>
  <p>Generated {escape(generated)} from the Windows benchmark folder at <code>{escape(str(ROOT))}</code>.</p>

  <section class="summary">
    <div class="tile">Embedding speed<strong>Ollama</strong><span>{ollama["embed_rate"]:.1f} chunks/s vs {turbo["embed_rate"]:.1f} chunks/s</span></div>
    <div class="tile">Eval speed<strong>TurboQuant</strong><span>{seconds_label(turbo["eval_seconds"])} vs {seconds_label(ollama["eval_seconds"])}</span></div>
    <div class="tile">Accuracy winner<strong>Ollama</strong><span>Higher on all four aggregate retrieval metrics</span></div>
    <div class="tile">Index size<strong>Tie</strong><span>{escape(ollama["index_size"])} each</span></div>
  </section>

  <div class="note">
    This run used the first 5,000 rows from the existing stratified 50K sample because the full 50K corpus would take roughly 2.5 hours per backend at the measured rate. Treat absolute recall as a 5K-slice result; the backend-to-backend deltas are the useful signal here.
  </div>

  <h2>Run Configuration</h2>
  <table>
    <thead><tr><th>Item</th><th>Value</th></tr></thead>
    <tbody>
      <tr><th>Corpus slice</th><td>5,000 chunks from <code>data/chunk_samples/stratified_50k.parquet</code></td></tr>
      <tr><th>Evaluation queries</th><td>{ollama["n_queries"]} queries</td></tr>
      <tr><th>Embedding dimension</th><td>4096</td></tr>
      <tr><th>TurboQuant server flags</th><td><code>--embedding --pooling last -ngl all -c 32768 -b 8192 -ub 8192 -fa on --cache-type-k turbo3 --cache-type-v turbo3</code></td></tr>
      <tr><th>TurboQuant model file</th><td><code>Qwen3-Embedding-8B-Q8_0.gguf</code> from <code>Qwen/Qwen3-Embedding-8B-GGUF</code></td></tr>
      <tr><th>Ollama model</th><td><code>qwen3-embedding:8b-q8_0</code></td></tr>
    </tbody>
  </table>

  <h2>Speed</h2>
  <table>
    <thead><tr><th>Metric</th><th>Ollama</th><th>TurboQuant</th><th>TurboQuant delta</th></tr></thead>
    <tbody>
      <tr><th>Embedding throughput</th><td>{ollama["embed_rate"]:.1f} chunks/s</td><td>{turbo["embed_rate"]:.1f} chunks/s</td><td>{speed_delta:+.1f}%</td></tr>
      <tr><th>Embedding wall time</th><td>{seconds_label(ollama["embed_seconds"])}</td><td>{seconds_label(turbo["embed_seconds"])}</td><td>{seconds_label(turbo["embed_seconds"] - ollama["embed_seconds"])}</td></tr>
      <tr><th>Eval wall time</th><td>{seconds_label(ollama["eval_seconds"])}</td><td>{seconds_label(turbo["eval_seconds"])}</td><td>{eval_delta:+.1f}%</td></tr>
      <tr><th>Estimated 50K embedding time</th><td>{seconds_label(50000 / ollama["embed_rate"])}</td><td>{seconds_label(50000 / turbo["embed_rate"])}</td><td>Based on final 5K rate</td></tr>
    </tbody>
  </table>

  <h2>Accuracy</h2>
  <table>
    <thead><tr><th>Metric</th><th>Ollama</th><th>TurboQuant</th><th>TurboQuant delta</th><th>Winner</th></tr></thead>
    <tbody>
      {metric_row("Vector recall@10", "vec_recall_10", ollama, turbo)}
      {metric_row("Vector MRR@10", "vec_mrr_10", ollama, turbo)}
      {metric_row("End-to-end recall@10", "e2e_recall_10", ollama, turbo)}
      {metric_row("End-to-end MRR@10", "e2e_mrr_10", ollama, turbo)}
    </tbody>
  </table>

  <h2>Accuracy By Source Type</h2>
  <table>
    <thead><tr><th>Source type</th><th>Ollama</th><th>TurboQuant</th><th>TurboQuant delta</th></tr></thead>
    <tbody>
      {source_rows(ollama, turbo)}
    </tbody>
  </table>

  <h2>Size</h2>
  <table>
    <thead><tr><th>Item</th><th>Ollama</th><th>TurboQuant</th></tr></thead>
    <tbody>
      <tr><th>Model payload</th><td>{escape(ollama["model_size"])}</td><td>{escape(turbo["model_size"])}</td></tr>
      <tr><th>Lance index for 5K embeddings</th><td>{escape(ollama["index_size"])}</td><td>{escape(turbo["index_size"])}</td></tr>
      <tr><th>TurboQuant Windows package</th><td>Not applicable</td><td>0.77 GB zip, 1.48 GB extracted runtime</td></tr>
    </tbody>
  </table>

  <h2>VRAM</h2>
  <table>
    <thead><tr><th>Item</th><th>Observed / estimated value</th></tr></thead>
    <tbody>
      <tr><th>TurboQuant model buffer on GPU</th><td>7,668.64 MiB</td></tr>
      <tr><th>TurboQuant turbo3 KV cache at 32K context</th><td>{turbo_kv_mib:,.0f} MiB total: 450 MiB K, 450 MiB V</td></tr>
      <tr><th>Estimated FP16 KV cache for same 32K context</th><td>{fp16_kv_mib:,.0f} MiB</td></tr>
      <tr><th>Estimated KV-cache saving</th><td>{kv_saved_mib:,.0f} MiB, about {fp16_kv_mib / turbo_kv_mib:.1f}x smaller than FP16 KV</td></tr>
      <tr><th>Ollama comparison caveat</th><td>Ollama did not expose a comparable per-buffer VRAM breakdown in this run, so this is a confirmed TurboQuant KV-cache advantage, not a measured total-process VRAM win over Ollama.</td></tr>
    </tbody>
  </table>

  <h2>Conclusion</h2>
  <p><span class="ok">Ollama wins embedding throughput and aggregate retrieval accuracy in this bounded Windows comparison.</span> TurboQuant ran successfully with Qwen's embedding settings and TurboQuant cache enabled, and its query/eval pass was faster, but the corpus embedding pass did not produce a speed advantage over Ollama. TurboQuant did provide a clear KV-cache VRAM advantage at 32K context; total VRAM was not captured apples-to-apples against Ollama. Size is effectively tied for the generated vector indexes; TurboQuant adds its standalone Windows runtime footprint.</p>
</main>
</body>
</html>
"""


def main() -> None:
    ollama = load_result("ollama")
    turbo = load_result("turbo")
    out = ROOT / "reports" / f"{REPORT_DATE}_qwen3_q8_turboquant_vs_ollama.html"
    out.write_text(html_report(ollama, turbo), encoding="utf-8")
    summary = {
        "report": str(out),
        "ollama": ollama,
        "turbo": turbo,
    }
    summary_out = ROOT / "reports" / f"{REPORT_DATE}_qwen3_q8_turboquant_vs_ollama_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out)
    print(summary_out)


if __name__ == "__main__":
    main()
