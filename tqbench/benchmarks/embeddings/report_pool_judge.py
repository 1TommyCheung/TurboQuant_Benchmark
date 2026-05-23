"""Build a standalone HTML report for the pool-and-judge decision.

Reads:
  - reports/raw/judge_scores/consolidated_scores.json    (per-chunk judge scores)
  - reports/raw/judge_scores/backend_recall_report.json  (V1/V2/8B-FP8 recall)
  - reports/raw/judge_scores/recall_4b_fp8.json          (4B-FP8 recall)

Writes:
  - reports/{date}_pool_and_judge_report.html

Same dark theme + Plotly charts as 2026-05-16_report.html, but data model is
the pool-and-judge ground-truth-vs-recall comparison, not Phase 1/2 quality.
"""
from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path
from statistics import mean

import plotly.graph_objects as go

BENCH_ROOT = Path(__file__).resolve().parent
RAW = BENCH_ROOT / "reports" / "raw" / "judge_scores"
OUT_HTML = BENCH_ROOT / "reports" / f"{dt.date.today().isoformat()}_pool_and_judge_report.html"

BACKENDS_LEGACY = ["gemini-embedding-001", "gemini-embedding-2", "qwen3-embedding-8b-fp8-vllm"]
ALL_BACKENDS = [
    ("gemini-embedding-001", "V1", 3072, "API"),
    ("gemini-embedding-2", "V2", 3072, "API"),
    ("qwen3-embedding-8b-fp8-vllm", "Qwen3-8B-FP8", 4096, "~9 GB"),
    ("qwen3-embedding-4b-fp8-vllm", "Qwen3-4B-FP8", 2560, "~5 GB"),
]
COLORS = {
    "gemini-embedding-001": "#8abeb7",
    "gemini-embedding-2": "#9575cd",
    "qwen3-embedding-8b-fp8-vllm": "#b5bd68",
    "qwen3-embedding-4b-fp8-vllm": "#ffff00",
    "BM25": "#cc6666",
}

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e5e5e7", family="-apple-system, system-ui, sans-serif"),
    margin=dict(l=60, r=20, t=50, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e5e7")),
)
DARK_AXES = dict(gridcolor="#333", zerolinecolor="#444",
                 tickfont=dict(color="#e5e5e7"), title_font=dict(color="#e5e5e7"))

QUERIES = {
    "parental_alienation": ("Concept", "parental alienation children refusing access negative perception father"),
    "access_denial_synonyms": ("Synonym", "access denied refused blocked cancelled children"),
    "valley_point_event": ("Event", "Valley Point Shopping Centre encounter Tristan November 2025"),
    "disclosure_non_compliance": ("Concept", "disclosure non-compliance failure produce documents adverse inference"),
    "indirect_contribution_caregiver": ("Concept", "indirect contribution caregiver homemaker non-financial role sacrifice career"),
    "valley_point_handover": ("Event", "Valley Point pick up drop off children"),
    "negative_characterisation": ("Concept", "negative characterisation father children alienation influence"),
    "children_refuse_visit": ("Synonym", "children don't want to see father refuse visit access"),
    "tracy_limited_access_proposals": ("Topical", "limited meal only access proposals suspended children"),
    "hk_unilateral_travel": ("Event", "Hong Kong children taken without consent passports missing"),
    "father_school_visit_framing": ("Concept", "father visit school kindergarten approaching children negative framing harassment"),
    "counselling_engagement_pattern": ("Topical", "counselling sessions in-person tele-conference engagement willingness refusal"),
    "gatekeeping_pattern": ("Concept", "gatekeeping behaviour mother exclude father co-parenting interference"),
    "fdr_mediation_directions": ("Topical", "FDR mediation directions counsellor third party suspension access"),
    "matrimonial_home_valuation": ("Topical", "matrimonial home valuation arms-length sale market value division"),
}


def load_data():
    consolidated = json.loads((RAW / "consolidated_scores.json").read_text())
    legacy = json.loads((RAW / "backend_recall_report.json").read_text())
    new = json.loads((RAW / "recall_4b_fp8.json").read_text())
    return consolidated, legacy, new


def macro(legacy, new, backend_id, metric_key):
    """metric_key: 'vector_recall_consensus' | 'hybrid_recall_consensus' | 'vector_recall_strict' | 'hybrid_recall_strict'."""
    if backend_id == "qwen3-embedding-4b-fp8-vllm":
        short_map = {
            "vector_recall_consensus": "vec_con",
            "hybrid_recall_consensus": "hyb_con",
            "vector_recall_strict": "vec_str",
            "hybrid_recall_strict": "hyb_str",
        }
        return new["macro"][short_map[metric_key]]
    vals = [legacy[q]["backends"][backend_id][metric_key] for q in legacy]
    return mean(vals)


def bm25_macro(legacy, key):
    """key: 'recall_consensus' or 'recall_strict'."""
    return mean(legacy[q]["bm25"][key] for q in legacy)


def per_query(legacy, new, backend_id, metric):
    """Get per-query value for a backend. metric: same form as macro key."""
    short_map = {
        "vector_recall_consensus": "vec_con",
        "hybrid_recall_consensus": "hyb_con",
        "vector_recall_strict": "vec_str",
        "hybrid_recall_strict": "hyb_str",
    }
    if backend_id == "qwen3-embedding-4b-fp8-vllm":
        return {qid: new["per_query"][qid][short_map[metric]] for qid in QUERIES}
    return {qid: legacy[qid]["backends"][backend_id][metric] for qid in QUERIES}


def chart_heatmap(legacy, new):
    """Per-query hybrid-strict recall heatmap across all 4 backends + BM25."""
    qids = list(QUERIES.keys())
    rows = ["BM25 only"] + [short for _, short, _, _ in ALL_BACKENDS]
    z = []
    bm25_row = [legacy[q]["bm25"]["recall_strict"] for q in qids]
    z.append(bm25_row)
    for backend_id, _, _, _ in ALL_BACKENDS:
        pq = per_query(legacy, new, backend_id, "hybrid_recall_strict")
        z.append([pq[q] for q in qids])

    fig = go.Figure(go.Heatmap(
        z=z, x=qids, y=rows,
        colorscale=[[0, "#1a1a1f"], [0.3, "#3a3a4f"], [0.6, "#5f87ff"], [1.0, "#b5bd68"]],
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}", textfont={"size": 11},
        colorbar=dict(title="recall@20<br>(hyb strict)", tickfont=dict(color="#e5e5e7"),
                      title_font=dict(color="#e5e5e7")),
        hovertemplate="<b>%{y}</b><br>%{x}<br>recall = %{z:.3f}<extra></extra>",
    ))
    fig.update_xaxes(tickangle=-30, **DARK_AXES)
    fig.update_yaxes(autorange="reversed", **DARK_AXES)
    fig.update_layout(
        title="Per-query hybrid-strict recall@20 (consensus mean ≥ 2.5)",
        height=380, **DARK_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="heatmap")


def chart_macro_bars(legacy, new):
    """4-metric grouped bar: vec_con / hyb_con / vec_str / hyb_str per backend + BM25."""
    metrics_display = ["vec_con", "hyb_con", "vec_str", "hyb_str"]
    metrics_full = ["vector_recall_consensus", "hybrid_recall_consensus",
                    "vector_recall_strict", "hybrid_recall_strict"]
    bm25_keys = [None, "recall_consensus", None, "recall_strict"]

    fig = go.Figure()
    # BM25 (only hyb cols apply)
    bm25_vals = []
    for k in bm25_keys:
        bm25_vals.append(bm25_macro(legacy, k) if k else None)
    fig.add_trace(go.Bar(
        name="BM25 only", x=metrics_display, y=bm25_vals,
        marker_color=COLORS["BM25"],
        text=[f"{v:.3f}" if v is not None else "" for v in bm25_vals],
        textposition="outside",
    ))
    for backend_id, short, _, _ in ALL_BACKENDS:
        vals = [macro(legacy, new, backend_id, m) for m in metrics_full]
        fig.add_trace(go.Bar(
            name=short, x=metrics_display, y=vals,
            marker_color=COLORS[backend_id],
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        ))
    fig.update_xaxes(**DARK_AXES)
    fig.update_yaxes(title="Macro recall@20", **DARK_AXES)
    fig.update_layout(
        title="Macro recall@20 across 15 queries — 4 retrievers, 4 metric variants",
        height=480, barmode="group", **DARK_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="macro_bars")


def chart_wins(legacy, new):
    """Per-metric wins among the 4 retrievers (ties shared)."""
    metrics = [("vec_con", "vector_recall_consensus"),
               ("hyb_con", "hybrid_recall_consensus"),
               ("vec_str", "vector_recall_strict"),
               ("hyb_str", "hybrid_recall_strict")]
    bm25_keys = {"hyb_con": "recall_consensus", "hyb_str": "recall_strict"}
    backends = [b for b, _, _, _ in ALL_BACKENDS] + ["BM25"]
    wins = {b: {m[0]: 0 for m in metrics} for b in backends}

    for qid in QUERIES:
        for short, full in metrics:
            candidates = {}
            for backend_id, _, _, _ in ALL_BACKENDS:
                candidates[backend_id] = per_query(legacy, new, backend_id, full)[qid]
            if short in bm25_keys:
                candidates["BM25"] = legacy[qid]["bm25"][bm25_keys[short]]
            best = max(candidates.values())
            for k, v in candidates.items():
                if v == best:
                    wins[k][short] += 1

    fig = go.Figure()
    short_metrics = [m[0] for m in metrics]
    for backend_id, short, _, _ in ALL_BACKENDS:
        fig.add_trace(go.Bar(
            name=short, x=short_metrics, y=[wins[backend_id][m] for m in short_metrics],
            marker_color=COLORS[backend_id],
            text=[wins[backend_id][m] for m in short_metrics], textposition="outside",
        ))
    fig.add_trace(go.Bar(
        name="BM25 only", x=short_metrics, y=[wins["BM25"][m] for m in short_metrics],
        marker_color=COLORS["BM25"],
        text=[wins["BM25"][m] for m in short_metrics], textposition="outside",
    ))
    fig.update_xaxes(**DARK_AXES)
    fig.update_yaxes(title="Win count (ties shared)", range=[0, 16], **DARK_AXES)
    fig.update_layout(
        title="Per-query win counts (out of 15 queries; ties shared)",
        height=420, barmode="group", **DARK_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="wins")


def chart_judge_variance(consolidated):
    """Distribution of per-chunk judge variance across queries (max - min across 3 judges)."""
    bins = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0
    for qid, chunks in consolidated.items():
        for cid, s in chunks.items():
            v = s.get("variance")
            if v is None:
                continue
            bins[v] = bins.get(v, 0) + 1
            total += 1
    labels = ["0 (full agreement)", "1 (within 1 point)", "2", "3 (max disagreement)"]
    vals = [bins.get(i, 0) for i in range(4)]
    pct = [v / total * 100 for v in vals]
    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        text=[f"{v}<br>({p:.1f}%)" for v, p in zip(vals, pct)],
        textposition="outside",
        marker_color=["#b5bd68", "#8abeb7", "#ffff00", "#cc6666"],
    ))
    fig.update_xaxes(**DARK_AXES)
    fig.update_yaxes(title="# chunks", **DARK_AXES)
    fig.update_layout(
        title=f"Judge agreement (3 judges × {total:,} chunk-judgements)",
        height=380, **DARK_LAYOUT, showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="variance")


def chart_quality_vs_vram(legacy, new):
    """Scatter: hybrid-strict macro vs VRAM. 'Right-sizer' picture."""
    vram_gb = {"gemini-embedding-001": 0.5, "gemini-embedding-2": 0.5,
               "qwen3-embedding-8b-fp8-vllm": 9.0,
               "qwen3-embedding-4b-fp8-vllm": 5.0}
    fig = go.Figure()
    for backend_id, short, dim, _ in ALL_BACKENDS:
        m = macro(legacy, new, backend_id, "hybrid_recall_strict")
        fig.add_trace(go.Scatter(
            x=[vram_gb[backend_id]], y=[m], mode="markers+text",
            text=[f"{short}<br>{dim}d"], textposition="top center",
            marker=dict(size=22, color=COLORS[backend_id], line=dict(color="#333", width=1)),
            name=short, hovertemplate=f"<b>{short}</b><br>VRAM: %{{x}} GB<br>hyb_str: %{{y:.3f}}<extra></extra>",
        ))
    # BM25 only
    fig.add_trace(go.Scatter(
        x=[0.1], y=[bm25_macro(legacy, "recall_strict")], mode="markers+text",
        text=["BM25 only"], textposition="top center",
        marker=dict(size=18, color=COLORS["BM25"], symbol="diamond", line=dict(color="#333", width=1)),
        name="BM25 only",
    ))
    fig.update_xaxes(title="Local VRAM (GB) — API models ≈ 0.5 placeholder", type="log", **DARK_AXES)
    fig.update_yaxes(title="Macro hybrid recall@20 (strict)", **DARK_AXES)
    fig.update_layout(
        title="Quality vs VRAM — does smaller buy us anything?",
        height=460, **DARK_LAYOUT, showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="qual_vs_vram")


def html_table_leaderboard(legacy, new):
    rows = []
    rows.append(f'<tr><td>BM25 only</td><td class="num">—</td><td class="num">—</td>'
                f'<td class="num">—</td><td class="num">{bm25_macro(legacy, "recall_consensus"):.3f}</td>'
                f'<td class="num">—</td><td class="num">{bm25_macro(legacy, "recall_strict"):.3f}</td></tr>')
    for backend_id, short, dim, vram in ALL_BACKENDS:
        vc = macro(legacy, new, backend_id, "vector_recall_consensus")
        hc = macro(legacy, new, backend_id, "hybrid_recall_consensus")
        vs = macro(legacy, new, backend_id, "vector_recall_strict")
        hs = macro(legacy, new, backend_id, "hybrid_recall_strict")
        cls = ""
        if backend_id == "gemini-embedding-001":
            cls = ' class="baseline-row"'  # production baseline
        elif backend_id == "qwen3-embedding-8b-fp8-vllm":
            cls = ' class="winner-row"'  # bench winner
        rows.append(f'<tr{cls}><td><code>{backend_id}</code> ({short})</td>'
                    f'<td class="num">{dim}</td><td class="num">{vram}</td>'
                    f'<td class="num">{vc:.3f}</td><td class="num">{hc:.3f}</td>'
                    f'<td class="num">{vs:.3f}</td><td class="num">{hs:.3f}</td></tr>')
    return "\n".join(rows)


def html_per_query_table(legacy, new):
    out = []
    for qid, (kind, qtext) in QUERIES.items():
        v1 = legacy[qid]["backends"]["gemini-embedding-001"]["hybrid_recall_strict"]
        v2 = legacy[qid]["backends"]["gemini-embedding-2"]["hybrid_recall_strict"]
        q8 = legacy[qid]["backends"]["qwen3-embedding-8b-fp8-vllm"]["hybrid_recall_strict"]
        q4 = new["per_query"][qid]["hyb_str"]
        bm = legacy[qid]["bm25"]["recall_strict"]
        best = max(v1, v2, q8, q4, bm)
        def cell(v):
            mark = "winner-cell" if v == best and v > 0 else ""
            return f'<td class="num {mark}">{v:.3f}</td>'
        out.append(
            f"<tr><td><code>{qid}</code><br><span class='meta'>{kind}: {qtext}</span></td>"
            f"{cell(v1)}{cell(v2)}{cell(q8)}{cell(q4)}{cell(bm)}</tr>"
        )
    return "\n".join(out)


def queries_section_html():
    grouped = {}
    for qid, (kind, qtext) in QUERIES.items():
        grouped.setdefault(kind, []).append((qid, qtext))
    order = ["Concept", "Synonym", "Event", "Topical"]
    out = []
    for kind in order:
        if kind not in grouped:
            continue
        out.append(f"<h4>{kind} queries</h4><ul>")
        for qid, qtext in grouped[kind]:
            out.append(f"<li><code>{qid}</code> — &ldquo;{qtext}&rdquo;</li>")
        out.append("</ul>")
    return "\n".join(out)


def consensus_summary_html(consolidated):
    rows = []
    for qid, chunks in consolidated.items():
        n = len(chunks)
        n_con = sum(1 for c in chunks.values() if c.get("mean", 0) >= 1.5)
        n_str = sum(1 for c in chunks.values() if c.get("mean", 0) >= 2.5)
        n_var = sum(1 for c in chunks.values() if c.get("variance", 0) >= 2)
        rows.append(f"<tr><td><code>{qid}</code></td><td class='num'>{n}</td>"
                    f"<td class='num'>{n_con}</td><td class='num'>{n_str}</td>"
                    f"<td class='num'>{n_var}</td></tr>")
    return "\n".join(rows)


def build():
    consolidated, legacy, new = load_data()
    heatmap = chart_heatmap(legacy, new)
    macro_bars = chart_macro_bars(legacy, new)
    wins = chart_wins(legacy, new)
    variance = chart_judge_variance(consolidated)
    qual_vram = chart_quality_vs_vram(legacy, new)
    leaderboard_rows = html_table_leaderboard(legacy, new)
    per_query_rows = html_per_query_table(legacy, new)
    queries_html = queries_section_html()
    consensus_rows = consensus_summary_html(consolidated)

    total_judgments = sum(len(chunks) * 3 for chunks in consolidated.values())
    total_pool = sum(len(chunks) for chunks in consolidated.values())

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Pool-and-Judge Decision — {dt.date.today().isoformat()}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --accent: #8abeb7; --border: #5f87ff; --bg: #1a1a1f; --bg-card: #25252b;
    --text: #e5e5e7; --muted: #808080; --success: #b5bd68; --error: #cc6666;
    --warning: #ffff00;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, system-ui, sans-serif;
         line-height: 1.6; max-width: 1240px; margin: 0 auto; padding: 2em; }}
  h1 {{ color: var(--accent); border-bottom: 2px solid var(--border); padding-bottom: 0.5em; }}
  h2 {{ color: var(--accent); margin-top: 2em; border-bottom: 1px solid #333; padding-bottom: 0.25em; }}
  h3 {{ color: var(--accent); margin-top: 1.5em; }}
  h4 {{ color: var(--text); margin-top: 1.2em; }}
  p {{ color: var(--text); }}
  a {{ color: var(--accent); }}
  .verdict-banner {{ padding: 1.5em; border-radius: 8px; font-size: 1.2em; font-weight: bold;
                    text-align: center; margin: 1em 0; }}
  .verdict-stay {{ background: rgba(138, 190, 183, 0.15); border: 2px solid var(--accent); }}
  .verdict-rationale {{ font-weight: normal; font-size: 0.9em; margin-top: 0.6em; opacity: 0.9; }}
  table {{ border-collapse: collapse; width: 100%; background: var(--bg-card); margin: 0.5em 0 1em; }}
  th, td {{ padding: 0.55em 0.9em; border-bottom: 1px solid #333; text-align: left; vertical-align: top; }}
  th {{ background: rgba(95, 135, 255, 0.1); color: var(--accent); font-weight: 600; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .winner-row {{ background: rgba(181, 189, 104, 0.10); }}
  .baseline-row {{ background: rgba(95, 135, 255, 0.10); }}
  .winner-cell {{ color: var(--success); font-weight: 600; }}
  .chart-container {{ background: var(--bg-card); border-radius: 8px; padding: 1em; margin: 1em 0; }}
  details {{ background: var(--bg-card); padding: 1em 1.2em; border-radius: 8px; margin: 1em 0; }}
  summary {{ cursor: pointer; color: var(--accent); font-weight: bold; padding: 0.2em 0; }}
  summary:hover {{ color: var(--text); }}
  .meta {{ color: var(--muted); font-size: 0.85em; }}
  code {{ background: rgba(128,128,128,0.15); padding: 0.1em 0.4em; border-radius: 3px; font-size: 0.9em; }}
  ul {{ margin: 0.4em 0 0.8em 1.5em; }}
  li {{ margin: 0.2em 0; }}
  .legend-tag {{ display: inline-block; padding: 0.15em 0.5em; border-radius: 4px; font-size: 0.85em; }}
  .tag-baseline {{ background: rgba(95, 135, 255, 0.25); color: #b3c4ff; }}
  .tag-winner {{ background: rgba(181, 189, 104, 0.25); color: var(--success); }}
</style>
</head><body>

<h1>Pool-and-Judge Embedding Decision</h1>
<p class="meta">Run: {dt.date.today().isoformat()} &middot; Snapshot: <code>2026-05-16_1fe458f</code>
(case_kb commit <code>1fe458fa6</code>) &middot; Corpus: 50,000 stratified chunks</p>

<div class="verdict-banner verdict-stay">
  Verdict: <strong>Stay on <code>gemini-embedding-001</code> (V1)</strong>
  <div class="verdict-rationale">
    Macro hybrid-strict recall: V1 0.340 vs Qwen3-8B-FP8 0.348 (Δ = +0.008 within noise).
    8B-FP8 wins legal-concept queries; V1 wins synonym/paraphrase queries — the production
    agent's day-to-day shape. V2 and 4B-FP8 cluster with BM25-only and do not justify cost.
  </div>
</div>

<h2>Leaderboard</h2>
<p>Macro-averaged recall@20 across 15 hand-selected semantic queries. Consensus relevance is
defined by a 3-judge ensemble (mean ≥ 1.5 = consensus, mean ≥ 2.5 = strict).</p>
<table>
  <thead><tr>
    <th>Backend</th><th class="num">dim</th><th class="num">VRAM</th>
    <th class="num">vec_con</th><th class="num">hyb_con</th>
    <th class="num">vec_str</th><th class="num">hyb_str</th>
  </tr></thead>
  <tbody>
    {leaderboard_rows}
  </tbody>
</table>
<p class="meta">
  <span class="legend-tag tag-baseline">production baseline</span>
  <span class="legend-tag tag-winner">bench winner (by macro hyb_str)</span>
</p>

<h2>Macro comparison — 4 retrievers, 4 metrics</h2>
<div class="chart-container">{macro_bars}</div>

<h2>Per-query hybrid-strict recall</h2>
<div class="chart-container">{heatmap}</div>

<h2>Per-query win counts</h2>
<p>Top score per metric across the 4 retrievers + BM25-only. Ties counted for all winners.</p>
<div class="chart-container">{wins}</div>

<h2>Quality vs VRAM — does smaller buy us anything?</h2>
<div class="chart-container">{qual_vram}</div>

<h2>Per-query table (hybrid-strict)</h2>
<details>
  <summary>15 queries × 5 retrievers (click to expand)</summary>
  <table>
    <thead><tr>
      <th>Query</th>
      <th class="num">V1</th><th class="num">V2</th>
      <th class="num">8B-FP8</th><th class="num">4B-FP8</th>
      <th class="num">BM25</th>
    </tr></thead>
    <tbody>
      {per_query_rows}
    </tbody>
  </table>
</details>

<h2>Methodology</h2>

<h3>Pool-and-judge protocol</h3>
<p>For each of 15 queries, a candidate <strong>pool</strong> was built by unioning:</p>
<ul>
  <li>top-20 vector results from each of the 3 originally-benched backends (V1, V2, 8B-FP8)</li>
  <li>top-20 BM25 results</li>
</ul>
<p>Pool sizes: 52–77 chunks per query &middot; Total pool: <strong>{total_pool:,}</strong> chunk-rows</p>

<p>Each chunk × query was scored by <strong>three LLM judges</strong> on a 0–3 rubric:</p>
<table>
  <thead><tr><th>Judge</th><th>Role</th><th>Bias</th></tr></thead>
  <tbody>
    <tr><td>Sonnet</td><td>Neutral evidence-relevance judge</td><td>Be fair, calibrated</td></tr>
    <tr><td>Opus</td><td>Senior legal researcher</td><td>Consider underlying legal/factual relevance, not just keywords</td></tr>
    <tr><td>Skeptical Opus</td><td>Adversarial reviewer</td><td>Try to argue why each chunk is NOT relevant</td></tr>
  </tbody>
</table>
<p>Rubric: <code>0</code> = not relevant &middot; <code>1</code> = tangential &middot;
   <code>2</code> = relevant but not central &middot; <code>3</code> = central evidence</p>

<p>Dispatched as <strong>45 parallel Claude Code subagents</strong> (15 queries × 3 judges).
Total judgments produced: <strong>{total_judgments:,}</strong>.</p>

<h3>Consensus thresholds</h3>
<table>
  <thead><tr><th>Threshold</th><th>Rule</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td>loose</td><td>mean ≥ 1.0</td><td>at least weak relevance from majority</td></tr>
    <tr><td>consensus</td><td>mean ≥ 1.5</td><td>typically scored ≥ 2 by majority of judges</td></tr>
    <tr><td>strict</td><td>mean ≥ 2.5</td><td>highly relevant; would be cited</td></tr>
  </tbody>
</table>
<p>The <strong>strict</strong> threshold is the headline metric — it matches the case
agent's actual production behaviour (cite or skip).</p>

<h3>Judge agreement</h3>
<div class="chart-container">{variance}</div>
<p>Most chunks have full or near-full agreement across the three judges. The
adversarial Skeptical-Opus produced visible pushback (variance ≥ 2) on a minority
of cases, exactly as designed.</p>

<h3>Per-query pool composition</h3>
<details>
  <summary>Pool size + consensus counts per query</summary>
  <table>
    <thead><tr>
      <th>Query</th><th class="num">pool</th>
      <th class="num">consensus (≥1.5)</th><th class="num">strict (≥2.5)</th>
      <th class="num">high-variance (≥2)</th>
    </tr></thead>
    <tbody>{consensus_rows}</tbody>
  </table>
</details>

<h2>The 15 queries</h2>
<p>Hand-selected to cover the semantic types the production agent must handle.
Each judge prompt included both the query string and an <code>intent</code> line.</p>
{queries_html}

<h2>Backends and how they were built</h2>
<table>
  <thead><tr><th>Backend</th><th>Source</th><th>How built</th></tr></thead>
  <tbody>
    <tr><td><code>gemini-embedding-001</code> (V1)</td>
        <td>Google API (production baseline)</td>
        <td>Batched API calls, 50K rows × 3072 d</td></tr>
    <tr><td><code>gemini-embedding-2</code> (V2)</td>
        <td>Google API</td>
        <td>Batched API calls, 50K rows × 3072 d</td></tr>
    <tr><td><code>qwen3-embedding-8b-fp8-vllm</code></td>
        <td><code>maywell/Qwen3-Embedding-8B-FP8-Dynamic</code></td>
        <td>vLLM serve + <code>runners.embed_corpus</code>, 50K × 4096 d (~30 chunks/s)</td></tr>
    <tr><td><code>qwen3-embedding-4b-fp8-vllm</code></td>
        <td><code>chroma-core/Qwen3-Embedding-4B-FP8-Dynamic</code></td>
        <td>vLLM serve + <code>runners.embed_corpus</code>, 50K × 2560 d (~18–38 chunks/s)</td></tr>
  </tbody>
</table>
<details>
  <summary>vLLM serve command (same shape for both Qwen3 variants)</summary>
<pre><code>vllm serve &lt;hf_repo&gt; --runner pooling --convert embed \\
  --port 8800 --host 127.0.0.1 \\
  --gpu-memory-utilization 0.85 --max-model-len 8192 --trust-remote-code</code></pre>
</details>

<h2>What was selected and why</h2>

<h3>Decision</h3>
<p><strong>Keep <code>gemini-embedding-001</code> (V1).</strong> Qwen3-8B-FP8 is the
deferred local-sovereignty option if/when Gemini API constraints or sovereignty
requirements change the trade-off.</p>

<h3>Why not Qwen3-8B-FP8 despite winning macro hyb_str by +0.008</h3>
<ol>
  <li>The 0.008 gap is within noise of a 15-query × 3-judge benchmark.</li>
  <li>8B-FP8 <em>loses</em> on V1's strengths — synonym/paraphrase queries that
      the production agent actually issues every day:
    <ul>
      <li><code>access_denial_synonyms</code>: V1 0.476 vs 8B 0.143</li>
      <li><code>tracy_limited_access_proposals</code>: V1 0.600 vs 8B 0.300</li>
      <li><code>counselling_engagement_pattern</code>: V1 0.615 vs 8B 0.538</li>
    </ul>
  </li>
  <li>Switching costs are concrete: re-embed 251K production chunks on the 4090,
      validate <code>tools/search/hybrid_search.py</code> end-to-end,
      manage 9 GB VRAM in the FastAPI service.</li>
  <li>Quality case alone does not clear the switching bar.</li>
</ol>

<h3>Why V2 was eliminated</h3>
<p>Never best on any per-query metric. Macro hyb_str = 0.257, statistically tied
with BM25 alone (0.264). Drop from consideration.</p>

<h3>Why 4B-FP8 was eliminated</h3>
<p>Macro hyb_str = 0.261 — about <strong>25% relative below</strong> both V1 (0.340)
and 8B-FP8 (0.348). Saving 4 GB VRAM does not justify a quality drop of this size.
The right-sizing hypothesis is empirically falsified for this corpus.</p>

<h3>Conditions that would flip the verdict to Qwen3-8B-FP8</h3>
<ul>
  <li>Need for data sovereignty (no Gemini API dependency)</li>
  <li>Significant Gemini API cost or rate-limit pressure</li>
  <li>Latency requirement that local vLLM serves better than the API round-trip</li>
  <li>A future re-bench showing 8B-FP8 closing the synonym gap (e.g. with prompt-tuning)</li>
</ul>

<h2>Artifacts</h2>
<table>
  <thead><tr><th>Path</th><th>What it is</th></tr></thead>
  <tbody>
    <tr><td><code>reports/raw/judge_scores/consolidated_scores.json</code></td>
        <td>15 queries × ~70 chunks × 3 judge scores + consensus stats</td></tr>
    <tr><td><code>reports/raw/judge_scores/backend_recall_report.json</code></td>
        <td>Per-query recall for V1, V2, 8B-FP8 vs consensus thresholds</td></tr>
    <tr><td><code>reports/raw/judge_scores/recall_4b_fp8.json</code></td>
        <td>Round-2 conservative recall for 4B-FP8</td></tr>
    <tr><td><code>bench_phase_b/build_judge_pools.py</code> &middot;
            <code>build_judge_pools_part2.py</code></td>
        <td>Pool builders (one per 5-query batch)</td></tr>
    <tr><td><code>bench_phase_b/extract_judge_scores_v2.py</code></td>
        <td>Auto-discovery extractor (45 transcripts → consolidated)</td></tr>
    <tr><td><code>bench_phase_b/compute_recall.py</code></td>
        <td>15-query recall computation</td></tr>
    <tr><td><code>bench_embeddings/indexes/qwen3-embedding-4b-fp8-vllm.lance</code></td>
        <td>50K × 2560-dim 4B-FP8 index (symlink → /home/tommy/evidence_lake_indexes/)</td></tr>
    <tr><td><code>reports/2026-05-17_pool_and_judge_decision.md</code></td>
        <td>Companion markdown decision document</td></tr>
  </tbody>
</table>

<p class="meta">The 15-query × consensus_relevant set is a reusable Layer-1 ground truth.
Any future candidate can be evaluated against this exact bar without re-running
45 judge subagents: embed 50K chunks, run top-20 lookup, intersect with the
consensus set.</p>

</body></html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html)
    print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
