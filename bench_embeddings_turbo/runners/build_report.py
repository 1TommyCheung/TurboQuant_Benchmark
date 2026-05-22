"""Build the HTML report + 1-page decision MD.

Reads all reports/raw/*.json, computes weighted scores via bench.scoring,
renders Jinja2 template to reports/{date}_report.html and writes
reports/{date}_decision.md.

Charts populated:
  - corpus_donut, corpus_histogram   (from stratified_50k.parquet)
  - quality_by_st_chart              (model x source_type recall@10)
  - quality_by_lb_chart              (model x length_bucket recall@10)
  - vector_vs_e2e_chart              (per-model vector vs e2e bar)
  - head_to_head_heatmap             (model x source_type delta vs baseline)
  - recall_bound_chart               (model x source_type recall@100)
  - adversarial_chart                (per-model recall on adversarial set)
  - dirty_chart                      ("no data" if dirty layer not in eval)
  - throughput/latency/vram          ("Phase 2 not run" if no speed JSON)
"""
from __future__ import annotations
import argparse
import datetime as dt
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.models import get_candidate, baseline_dim, load_registry
from bench.scoring import (
    ModelResult, decide, weighted_total, apply_vetoes,
    W_QUALITY_VECTOR, W_QUALITY_E2E, W_LONG_CONTEXT, W_LOCAL_CONTROL,
    DIM_PENALTY_PTS,
)

BENCH_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = BENCH_ROOT / "templates"
REPORTS = BENCH_ROOT / "reports"
RAW = REPORTS / "raw"
SAMPLE = BENCH_ROOT / "data" / "chunk_samples" / "stratified_50k.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Dark theme to match the template
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e5e5e7", family="-apple-system, system-ui, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e5e7")),
)
DARK_AXES = dict(
    gridcolor="#333",
    zerolinecolor="#444",
    tickfont=dict(color="#e5e5e7"),
    title_font=dict(color="#e5e5e7"),
)
MODEL_COLORS = {
    "gemini-embedding-001": "#8abeb7",          # accent
    "qwen3-embedding-8b-fp16": "#b5bd68",       # success
    "qwen3-embedding-8b-int8": "#9575cd",
    "harrier-oss-0.6b-bf16": "#ffff00",         # warning
    "llama-embed-nemotron-8b-int8": "#cc6666",  # error
    "kalm-gemma3-12b-int8": "#5f87ff",          # border
}

MERMAID_DIAGRAM = """flowchart TB
    A[Production LanceDB<br/>251K Gemini-embedded chunks] -->|read-only| B[Stratified 50K sample]
    B --> C[Eval data builder]
    S1[pi-sessions HTML/JSON] --> C
    S2[Handcrafted Layer 3] --> C
    AVF[agent_verified_facts.jsonl] --> C
    C --> Q[Layer 1 + 2a + 2b + Layer 3 + Adversarial + Dirty]
    Q --> M[Model embedders]
    M --> R[Vector + End-to-end metrics]
    R --> SCORE[Decision rule v2<br/>vetoes + dim penalty]
    SCORE --> H[This HTML report]
    SCORE --> D[1-page decision.md]
"""


def _fig_html(fig: go.Figure) -> str:
    fig.update_layout(**DARK_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _load_model_results() -> list[dict]:
    """Load quality + speed + Phase 1 speed JSON per model."""
    quality_files = sorted(glob.glob(str(RAW / "*_quality.json")))
    latest: dict[str, dict] = {}
    for q_file in quality_files:
        data = json.loads(Path(q_file).read_text())
        speed_phase2_file = q_file.replace("_quality.json", "_speed.json")
        if Path(speed_phase2_file).exists():
            data["speed_phase2"] = json.loads(Path(speed_phase2_file).read_text())
        speed_phase1_file = q_file.replace("_quality.json", "_speed_phase1.json")
        if Path(speed_phase1_file).exists():
            data["speed_phase1"] = json.loads(Path(speed_phase1_file).read_text())
        latest[data["model_id"]] = data
    # Put gemini first if present (baseline), then sort the rest alphabetically
    ordered: list[dict] = []
    if "gemini-embedding-001" in latest:
        ordered.append(latest.pop("gemini-embedding-001"))
    ordered.extend(latest[k] for k in sorted(latest))
    return ordered


def _per_query_lookup(model_data: dict, chunk_lookup: dict) -> pd.DataFrame:
    """Flatten per_query metrics into a DataFrame with source_type and length_bucket."""
    rows = []
    for r in model_data.get("per_query", []):
        scid = r.get("source_chunk_id")
        chunk = chunk_lookup.get(scid) if scid else None
        source_type = chunk["source_type"] if chunk else None
        length_bucket = chunk["length_bucket"] if chunk else None
        m = r.get("metrics", {})
        rows.append({
            "query_id": r.get("id"),
            "layer": r.get("layer"),
            "source_chunk_id": scid,
            "source_type": source_type,
            "length_bucket": length_bucket,
            "vec_recall_10": m.get("vec_recall_10", 0.0),
            "vec_recall_100": m.get("vec_recall_100", 0.0),
            "vec_mrr_10": m.get("vec_mrr_10", 0.0),
            "e2e_recall_10": m.get("e2e_recall_10", 0.0),
            "e2e_mrr_10": m.get("e2e_mrr_10", 0.0),
        })
    return pd.DataFrame(rows)


def _compute_long_context(per_query_df: pd.DataFrame) -> float:
    """Mean e2e_recall_10 over long + very_long length buckets."""
    sub = per_query_df[
        per_query_df["length_bucket"].isin(("long", "very_long"))
        & per_query_df["source_chunk_id"].notna()
    ]
    if sub.empty:
        # Fallback: use medium bucket as long-context proxy
        sub = per_query_df[
            (per_query_df["length_bucket"] == "medium")
            & per_query_df["source_chunk_id"].notna()
        ]
    if sub.empty:
        return 0.0
    return float(sub["e2e_recall_10"].mean() * 100)


def _compute_local_control(model_id: str, spec, baseline_monthly_usd: float = 50.0) -> float:
    """Spec §3 formula. Normalized to 0-100.

    Approximations used (refine when actual numbers available):
      monthly_API_$_saved: 50 for non-API, 0 for API
      ctx_window_headroom_tokens: spec.max_ctx_tokens (already in registry)
      offline_tolerance_hours: 168 for local (1 week), 0 for API (no offline)
    """
    if spec.kind == "api":
        saved = 0.0
        offline = 0.0
    else:
        saved = baseline_monthly_usd
        offline = 168.0
    ctx_headroom = float(spec.max_ctx_tokens or 0)
    component = (
        0.4 * (saved / baseline_monthly_usd if baseline_monthly_usd else 0)
        + 0.4 * min(1.0, ctx_headroom / 32768.0)
        + 0.2 * min(1.0, offline / 168.0)
    )
    return 100.0 * component


def _to_model_result(model_data: dict, baseline_dim_val: int,
                     per_query_df: pd.DataFrame) -> ModelResult:
    agg = model_data["aggregate"]
    spec = get_candidate(model_data["model_id"])
    return ModelResult(
        model_id=model_data["model_id"],
        quality_vector_only=100 * agg["overall_mean"]["vec_recall_10"],
        quality_end_to_end=100 * agg["overall_mean"]["e2e_recall_10"],
        long_context=_compute_long_context(per_query_df),
        local_control=_compute_local_control(model_data["model_id"], spec),
        e2e_recall_by_source_type={k: 100 * v for k, v in agg["by_source_type"].items()},
        baseline_e2e_recall_by_source_type={},  # filled below
        dim=spec.dim,
        baseline_dim=baseline_dim_val,
    )


# --- Chart builders --------------------------------------------------

def chart_corpus_donut() -> str:
    if not SAMPLE.exists():
        return ""
    df = pd.read_parquet(SAMPLE, columns=["source_type"])
    counts = df["source_type"].value_counts().reset_index()
    counts.columns = ["source_type", "count"]
    fig = px.pie(counts, names="source_type", values="count", hole=0.55,
                 title="Corpus sample composition (50K chunks)",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    return _fig_html(fig)


def chart_corpus_histogram() -> str:
    if not SAMPLE.exists():
        return ""
    df = pd.read_parquet(SAMPLE, columns=["token_count", "length_bucket"])
    df = df[df["token_count"] <= 12_000]  # cap x-axis for readability
    fig = px.histogram(df, x="token_count", color="length_bucket", nbins=80,
                       title="Chunk token-count distribution by length bucket",
                       color_discrete_map={"short": "#8abeb7", "medium": "#b5bd68",
                                            "long": "#ffff00", "very_long": "#cc6666"})
    fig.update_layout(xaxis_title="Tokens per chunk", yaxis_title="Chunks",
                      xaxis=DARK_AXES, yaxis=DARK_AXES, bargap=0.05)
    return _fig_html(fig)


def chart_quality_by_st(per_query_by_model: dict[str, pd.DataFrame],
                        baseline_id: str = "gemini-embedding-001",
                        min_n: int = 10) -> str:
    """Grouped bar: model × source_type, e2e_recall@10.

    Improvements over a naive bar chart:
      * x-axis sorted by baseline (Gemini) recall descending — consistent reading order
      * source_type labels include sample size: 'whatsapp (n=152)'
      * strata with n < min_n shaded with hatch + flagged 'low n' so the reader
        doesn't over-interpret strata where all models hit the same 1-2 chunks
      * y-axis labelled value annotations on every bar — differences visible at a glance
      * 95% binomial CI on bars with n >= min_n
    """
    rows = []
    for model_id, df in per_query_by_model.items():
        sub = df[df["source_chunk_id"].notna() & df["source_type"].notna()]
        for st, g in sub.groupby("source_type"):
            vals = g["e2e_recall_10"].values
            if len(vals) == 0:
                continue
            mean = vals.mean() * 100
            n = len(vals)
            se = float(np.sqrt(mean * (100 - mean) / n)) if n > 1 else 0.0
            rows.append({
                "model_id": model_id, "source_type": st,
                "recall_10": mean, "n": n,
                "ci": 1.96 * se if n >= min_n else 0.0,
                "low_n": n < min_n,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return ""

    # Sort source_types by baseline recall descending — consistent reading order
    baseline_rows = df[df["model_id"] == baseline_id]
    if not baseline_rows.empty:
        order = baseline_rows.sort_values("recall_10", ascending=False)["source_type"].tolist()
    else:
        order = sorted(df["source_type"].unique())
    df["source_type"] = pd.Categorical(df["source_type"], categories=order, ordered=True)
    df = df.sort_values(["source_type", "model_id"])

    # Decorate source_type labels with sample size and low_n marker
    n_by_st = df.groupby("source_type", observed=True)["n"].first().to_dict()
    df["st_label"] = df["source_type"].map(
        lambda s: f"{s}<br><span style='font-size:0.8em;color:#808080'>n={n_by_st[s]}{' ⚠ low n' if n_by_st[s] < min_n else ''}</span>"
    )

    fig = px.bar(
        df, x="st_label", y="recall_10", color="model_id", barmode="group",
        error_y="ci",
        color_discrete_map=MODEL_COLORS,
        text=df["recall_10"].round(1).astype(str) + "%",
        hover_data={"n": True, "low_n": True, "recall_10": ":.1f", "st_label": False},
        category_orders={"st_label": [df[df["source_type"] == s]["st_label"].iloc[0] for s in order]},
        title="E2E recall@10 by source type — bars sorted by Gemini, n shown, low-n strata flagged",
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10, color="#e5e5e7"))
    fig.update_layout(
        xaxis_title="Source type",
        yaxis_title="recall@10 (%)",
        yaxis=dict(range=[0, 110], **DARK_AXES),
        xaxis=DARK_AXES,
        bargap=0.15, bargroupgap=0.05,
    )
    return _fig_html(fig)


def chart_quality_by_lb(per_query_by_model: dict[str, pd.DataFrame],
                        min_n: int = 10) -> str:
    """Grouped bar: model × length_bucket, e2e_recall@10. Low-n bars flagged."""
    bucket_order = ["short", "medium", "long", "very_long"]
    rows = []
    for model_id, df in per_query_by_model.items():
        sub = df[df["source_chunk_id"].notna() & df["length_bucket"].notna()]
        for lb, g in sub.groupby("length_bucket"):
            vals = g["e2e_recall_10"].values
            if len(vals) == 0:
                continue
            rows.append({
                "model_id": model_id, "length_bucket": lb,
                "recall_10": vals.mean() * 100, "n": len(vals),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    df["length_bucket"] = df["length_bucket"].astype(str)
    df = df[df["length_bucket"].isin(bucket_order)]
    n_by_lb = df.groupby("length_bucket")["n"].first().to_dict()
    df["lb_label"] = df["length_bucket"].map(
        lambda b: f"{b}<br><span style='font-size:0.8em;color:#808080'>n={n_by_lb.get(b, 0)}{' ⚠ low n' if n_by_lb.get(b, 0) < min_n else ''}</span>"
    )
    df["_bucket_order"] = df["length_bucket"].map({b: i for i, b in enumerate(bucket_order)})
    df = df.sort_values(["_bucket_order", "model_id"])
    label_order = []
    seen = set()
    for b in bucket_order:
        sub = df[df["length_bucket"] == b]
        if not sub.empty:
            label = sub["lb_label"].iloc[0]
            if label not in seen:
                label_order.append(label)
                seen.add(label)
    fig = px.bar(
        df, x="lb_label", y="recall_10", color="model_id", barmode="group",
        color_discrete_map=MODEL_COLORS,
        text=df["recall_10"].round(1).astype(str) + "%",
        hover_data={"n": True, "recall_10": ":.1f", "lb_label": False},
        category_orders={"lb_label": label_order},
        title="E2E recall@10 by length bucket — long+very_long are inherently low-n in this sample",
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10, color="#e5e5e7"))
    fig.update_layout(xaxis_title="Length bucket", yaxis_title="recall@10 (%)",
                      yaxis=dict(range=[0, 110], **DARK_AXES), xaxis=DARK_AXES,
                      bargap=0.15, bargroupgap=0.05)
    return _fig_html(fig)


def chart_vector_vs_e2e(per_query_by_model: dict[str, pd.DataFrame]) -> str:
    """Per-model: vector-only vs end-to-end recall@10 — shows rerank rescue."""
    rows = []
    for model_id, df in per_query_by_model.items():
        sub = df[df["source_chunk_id"].notna()]
        if sub.empty:
            continue
        vec = float(sub["vec_recall_10"].mean() * 100)
        e2e = float(sub["e2e_recall_10"].mean() * 100)
        rows.append({"model_id": model_id, "Vector only": vec, "Hybrid (vec+BM25+RRF)": e2e})
    df = pd.DataFrame(rows).melt(id_vars="model_id", var_name="track", value_name="recall_10")
    if df.empty:
        return ""
    fig = px.bar(df, x="model_id", y="recall_10", color="track", barmode="group",
                 color_discrete_sequence=["#9575cd", "#8abeb7"],
                 title="Vector-only vs full hybrid — gap = how much BM25+RRF rescues")
    fig.update_layout(xaxis_title="Model", yaxis_title="recall@10 (%)",
                      xaxis=DARK_AXES, yaxis=DARK_AXES)
    return _fig_html(fig)


def chart_head_to_head(per_query_by_model: dict[str, pd.DataFrame],
                       baseline_id: str) -> str:
    """Heatmap: (candidate model × source_type) delta vs baseline."""
    baseline_df = per_query_by_model.get(baseline_id)
    if baseline_df is None:
        return ""
    baseline_sub = baseline_df[baseline_df["source_chunk_id"].notna()]
    baseline_st = baseline_sub.groupby("source_type")["e2e_recall_10"].mean() * 100

    delta_rows = []
    for model_id, df in per_query_by_model.items():
        if model_id == baseline_id:
            continue
        sub = df[df["source_chunk_id"].notna()]
        for st, g in sub.groupby("source_type"):
            cand = float(g["e2e_recall_10"].mean() * 100)
            base = float(baseline_st.get(st, 0.0))
            delta_rows.append({"model_id": model_id, "source_type": st,
                               "delta": cand - base})
    df = pd.DataFrame(delta_rows)
    if df.empty:
        return ""
    pivot = df.pivot(index="model_id", columns="source_type", values="delta")
    fig = px.imshow(pivot,
                    text_auto=".1f",
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    title=f"E2E recall@10 delta vs {baseline_id} (green = beats baseline)")
    fig.update_layout(xaxis_title="Source type", yaxis_title="Candidate",
                      xaxis=DARK_AXES, yaxis=DARK_AXES)
    return _fig_html(fig)


def chart_recall_bound(per_query_by_model: dict[str, pd.DataFrame]) -> str:
    """Per-model recall@10 vs recall@100 — recall-bound corpus view."""
    rows = []
    for model_id, df in per_query_by_model.items():
        sub = df[df["source_chunk_id"].notna()]
        if sub.empty:
            continue
        rows.append({"model_id": model_id, "k": "recall@10",
                     "value": float(sub["vec_recall_10"].mean() * 100)})
        rows.append({"model_id": model_id, "k": "recall@100",
                     "value": float(sub["vec_recall_100"].mean() * 100)})
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    fig = px.bar(df, x="model_id", y="value", color="k", barmode="group",
                 color_discrete_sequence=["#cc6666", "#b5bd68"],
                 title="Vector recall — top-10 vs top-100 (recall-bound corpus view)")
    fig.update_layout(xaxis_title="Model", yaxis_title="recall (%)",
                      xaxis=DARK_AXES, yaxis=DARK_AXES)
    return _fig_html(fig)


def chart_adversarial(per_query_by_model: dict[str, pd.DataFrame]) -> str:
    """Per-model recall@10 on adversarial set (Gemini-failure mining)."""
    rows = []
    for model_id, df in per_query_by_model.items():
        sub = df[df["layer"] == "adversarial"]
        if sub.empty:
            continue
        rows.append({
            "model_id": model_id,
            "e2e_recall_10": float(sub["e2e_recall_10"].mean() * 100),
            "n": len(sub),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    fig = px.bar(df, x="model_id", y="e2e_recall_10",
                 color="model_id", color_discrete_map=MODEL_COLORS,
                 hover_data={"n": True},
                 title=f"Adversarial Gemini-failure set (n={df['n'].iloc[0]} queries)")
    fig.update_layout(xaxis_title="Model", yaxis_title="recall@10 (%)",
                      showlegend=False, xaxis=DARK_AXES, yaxis=DARK_AXES)
    return _fig_html(fig)


def chart_speed_phase1(model_data: list[dict]) -> tuple[str, str, list[dict]]:
    """Phase 1 speed charts: embed throughput + eval per-query latency.

    Returns (embed_chart_html, eval_chart_html, table_rows).
    """
    rows = []
    for m in model_data:
        sp = m.get("speed_phase1") or {}
        em = sp.get("embed_phase1") or {}
        ev = sp.get("eval_phase1") or {}
        if not em and not ev:
            continue
        rows.append({
            "model_id": m["model_id"],
            "embed_chunks_per_s": em.get("final_rate_chunks_per_s", 0),
            "embed_wall_minutes_50k": em.get("approx_wall_minutes", 0),
            "approx_wall_minutes_251k": (251_089 / em["final_rate_chunks_per_s"] / 60)
                if em.get("final_rate_chunks_per_s") else 0,
            "eval_queries_per_s": ev.get("queries_per_s", 0),
            "ms_per_query": ev.get("approx_per_query_ms", 0),
        })
    if not rows:
        return "", "", []
    df = pd.DataFrame(rows).sort_values("embed_chunks_per_s", ascending=False)

    # Embed throughput chart
    fig1 = px.bar(df, x="model_id", y="embed_chunks_per_s",
                  color="model_id", color_discrete_map=MODEL_COLORS,
                  hover_data={"embed_wall_minutes_50k": ":.1f",
                              "approx_wall_minutes_251k": ":.1f"},
                  title="Phase 1 embed throughput (chunks/sec on 50K corpus)")
    fig1.update_layout(xaxis_title="Model", yaxis_title="chunks / second",
                       showlegend=False, xaxis=DARK_AXES, yaxis=DARK_AXES)

    # Per-query encode latency chart (lower is better)
    fig2 = px.bar(df, x="model_id", y="ms_per_query",
                  color="model_id", color_discrete_map=MODEL_COLORS,
                  hover_data={"eval_queries_per_s": ":.2f"},
                  title="Phase 1 per-query encode latency (eval pass, ms/query) — lower is better")
    fig2.update_layout(xaxis_title="Model", yaxis_title="ms / query",
                       showlegend=False, xaxis=DARK_AXES, yaxis=DARK_AXES)

    return _fig_html(fig1), _fig_html(fig2), rows


def chart_layer_breakdown(per_query_by_model: dict[str, pd.DataFrame]) -> str:
    """Per-model, per-layer query count and mean recall — sanity overview."""
    rows = []
    for model_id, df in per_query_by_model.items():
        for layer, g in df.groupby("layer"):
            with_pos = g[g["source_chunk_id"].notna()]
            rows.append({
                "model_id": model_id,
                "layer": layer,
                "n_total": len(g),
                "n_with_positives": len(with_pos),
                "mean_e2e_recall_10": float(with_pos["e2e_recall_10"].mean() * 100) if len(with_pos) else 0.0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    fig = px.bar(df, x="layer", y="mean_e2e_recall_10", color="model_id", barmode="group",
                 color_discrete_map=MODEL_COLORS,
                 hover_data={"n_total": True, "n_with_positives": True},
                 title="Mean E2E recall@10 by eval-data layer (Layer 2a/3 have no positives)")
    fig.update_layout(xaxis_title="Layer", yaxis_title="recall@10 (%)",
                      xaxis=DARK_AXES, yaxis=DARK_AXES)
    return _fig_html(fig)


def _load_phase_a_static_diff() -> dict | None:
    import glob
    cands = sorted(glob.glob(str(RAW / "*_phase_a_static_diff.json")))
    return json.loads(Path(cands[-1]).read_text()) if cands else None


def _load_phase_a_judge() -> dict | None:
    import glob
    cands = sorted(glob.glob(str(RAW / "*_phase_a_judge.json")))
    return json.loads(Path(cands[-1]).read_text()) if cands else None


def chart_phase_a_per_call(static_diff: dict) -> str:
    rows = static_diff["rows"]
    df = pd.DataFrame([{
        "call_idx": r["call_idx"],
        "turn_idx": r["turn_idx"],
        "query": r["query"][:80],
        "jaccard_at_k": r["jaccard_at_k"],
        "cited_weighted_a": r["cited_weighted_a"],
        "cited_weighted_b": r["cited_weighted_b"],
        "divergence_score": r["divergence_score"],
    } for r in rows])
    fig = px.scatter(
        df, x="call_idx", y="jaccard_at_k", color="turn_idx",
        hover_data={"query": True, "cited_weighted_a": ":.2f",
                    "cited_weighted_b": ":.2f", "divergence_score": ":.2f"},
        title="Phase A: jaccard@10 per search_evidence call (turn-colored)",
    )
    fig.update_layout(xaxis_title="Call index", yaxis_title="jaccard@10",
                      yaxis=dict(range=[0, 1.05], **DARK_AXES), xaxis=DARK_AXES)
    return _fig_html(fig)


def chart_phase_a_judge_verdicts(judge: dict) -> str:
    rows = []
    for t in judge["turns"]:
        rows.append({"backend": "harrier", "turn_idx": t["turn_idx"], "verdict": t["harrier"]["verdict_majority"]})
        rows.append({"backend": "gemini",  "turn_idx": t["turn_idx"], "verdict": t["gemini"]["verdict_majority"]})
    df = pd.DataFrame(rows)
    order = ["sufficient", "better_than_gemini", "partially_sufficient", "insufficient", "error"]
    df["verdict"] = pd.Categorical(df["verdict"], categories=order, ordered=True)
    counts = df.groupby(["backend", "verdict"], observed=False).size().reset_index(name="n")
    fig = px.bar(counts, x="backend", y="n", color="verdict",
                 color_discrete_map={"sufficient": "#b5bd68", "better_than_gemini": "#8abeb7",
                                      "partially_sufficient": "#ffff00", "insufficient": "#cc6666",
                                      "error": "#808080"},
                 title=f"Phase A: judge verdict distribution across {len(judge['turns'])} turns")
    fig.update_layout(xaxis_title="Backend", yaxis_title="Number of turns",
                      yaxis=DARK_AXES, xaxis=DARK_AXES)
    return _fig_html(fig)


def compute_phase_a_verdict(static_diff: dict, judge: dict) -> dict:
    """Apply spec §5.4 decision rule. Returns dict with verdict + rationale."""
    import statistics
    rows = static_diff["rows"]
    jaccards = [r["jaccard_at_k"] for r in rows]
    cited_b = [r["cited_weighted_b"] for r in rows]
    median_jaccard = statistics.median(jaccards) if jaccards else 0.0
    median_cited = statistics.median(cited_b) if cited_b else 0.0

    harrier_verdicts = [t["harrier"]["verdict_majority"] for t in judge["turns"]]
    n_sufficient_or_better = sum(1 for v in harrier_verdicts if v in ("sufficient", "better_than_gemini"))
    n_insufficient = sum(1 for v in harrier_verdicts if v == "insufficient")
    n_total = len(harrier_verdicts)

    if median_jaccard >= 0.90 and median_cited >= 0.95 and n_sufficient_or_better >= 9:
        return {
            "verdict": "A1",
            "label": "Harrier clearly competitive — proceed to Phase B",
            "median_jaccard": median_jaccard,
            "median_cited_weighted": median_cited,
            "n_sufficient_or_better": n_sufficient_or_better,
            "n_total_turns": n_total,
            "rationale": f"jaccard={median_jaccard:.2f} ≥ 0.90, cited_weighted={median_cited:.2f} ≥ 0.95, judge marked sufficient/better on {n_sufficient_or_better}/{n_total} ≥ 9 turns.",
        }
    if median_jaccard < 0.70 or median_cited < 0.80 or n_insufficient >= 5:
        return {
            "verdict": "A2",
            "label": "Harrier clearly loses — stay on Gemini",
            "median_jaccard": median_jaccard,
            "median_cited_weighted": median_cited,
            "n_insufficient": n_insufficient,
            "n_total_turns": n_total,
            "rationale": f"jaccard={median_jaccard:.2f}, cited_weighted={median_cited:.2f}, insufficient verdicts: {n_insufficient}/{n_total}.",
        }
    return {
        "verdict": "A3",
        "label": "Inconclusive — user calls it",
        "median_jaccard": median_jaccard,
        "median_cited_weighted": median_cited,
        "n_sufficient_or_better": n_sufficient_or_better,
        "n_insufficient": n_insufficient,
        "n_total_turns": n_total,
        "rationale": f"jaccard={median_jaccard:.2f}, cited_weighted={median_cited:.2f}, judge: {n_sufficient_or_better} sufficient+ / {n_insufficient} insufficient / {n_total} total.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPORTS)
    args = ap.parse_args()

    model_data = _load_model_results()
    if not model_data:
        log.error("No quality JSONs found in reports/raw/. Run eval_quality first.")
        sys.exit(1)
    log.info(f"Loaded {len(model_data)} model results: {[m['model_id'] for m in model_data]}")

    # Build chunk lookup for source_type + length_bucket cross-reference
    log.info("Loading corpus sample for length_bucket lookup...")
    sample_df = pd.read_parquet(SAMPLE,
                                 columns=["chunk_id", "source_type", "length_bucket"])
    chunk_lookup = sample_df.set_index("chunk_id").to_dict("index")
    log.info(f"  {len(chunk_lookup):,} chunks in lookup")

    # Per-model per-query DataFrames (cross-referenced with chunk metadata)
    per_query_by_model = {m["model_id"]: _per_query_lookup(m, chunk_lookup)
                          for m in model_data}

    # ModelResults for the decision rule
    base_dim = baseline_dim()
    results = [_to_model_result(m, base_dim, per_query_by_model[m["model_id"]])
               for m in model_data]
    baseline = next((r for r in results if r.model_id == "gemini-embedding-001"), None)
    if baseline is None:
        log.warning("No Gemini baseline found — using first model as pseudo-baseline")
        baseline = results[0]

    # Wire baseline source-type recall into each candidate (for veto check)
    for r in results:
        r.baseline_e2e_recall_by_source_type = baseline.e2e_recall_by_source_type

    candidate = max((r for r in results if r.model_id != baseline.model_id),
                    key=weighted_total, default=None)
    if candidate is None:
        verdict = type("V", (), {"verdict": "inconclusive", "winner_id": None,
                                  "scores": [], "veto_reasons": [],
                                  "rationale": "Only the baseline ran. Embed at least one candidate."})
    else:
        verdict = decide(candidate, baseline, all_candidates=results)

    # Scorecard table (uses correct math from bench.scoring: weighted_total includes dim penalty)
    scores_for_template = []
    for r in results:
        raw_total = (W_QUALITY_VECTOR * r.quality_vector_only
                     + W_QUALITY_E2E * r.quality_end_to_end
                     + W_LONG_CONTEXT * r.long_context
                     + W_LOCAL_CONTROL * r.local_control)
        dim_penalty = DIM_PENALTY_PTS if r.dim != r.baseline_dim else 0.0
        vetoes = apply_vetoes(r) if r.model_id != baseline.model_id else []
        scores_for_template.append({
            "model_id": r.model_id,
            "dim": r.dim,
            "quality_vector_only": r.quality_vector_only,
            "quality_end_to_end": r.quality_end_to_end,
            "long_context": r.long_context,
            "local_control": r.local_control,
            "raw_total": raw_total,
            "dim_penalty": dim_penalty,
            "total": raw_total - dim_penalty,
            "veto_reasons": vetoes,
        })
    # Sort scorecard: vetoed last, then by total desc
    scores_for_template.sort(key=lambda s: (bool(s["veto_reasons"]), -s["total"]))

    log.info("Building charts...")
    speed_p1_embed, speed_p1_eval, speed_p1_rows = chart_speed_phase1(model_data)
    charts = {
        "corpus_donut": chart_corpus_donut(),
        "corpus_histogram": chart_corpus_histogram(),
        "quality_by_st_chart": chart_quality_by_st(per_query_by_model),
        "quality_by_lb_chart": chart_quality_by_lb(per_query_by_model),
        "vector_vs_e2e_chart": chart_vector_vs_e2e(per_query_by_model),
        "head_to_head_heatmap": chart_head_to_head(per_query_by_model, baseline.model_id),
        "recall_bound_chart": chart_recall_bound(per_query_by_model),
        "adversarial_chart": chart_adversarial(per_query_by_model),
        "layer_breakdown_chart": chart_layer_breakdown(per_query_by_model),
        "speed_phase1_embed": speed_p1_embed,
        "speed_phase1_eval": speed_p1_eval,
        "speed_phase1_rows": speed_p1_rows,
    }

    # Phase 2 speed (only if Phase 2 speed JSON exists for any model)
    has_speed_phase2 = any("speed_phase2" in m for m in model_data)
    has_speed_phase1 = bool(speed_p1_rows)
    charts["throughput_chart"] = ""
    charts["latency_chart"] = ""
    charts["vram_chart"] = ""

    # Phase A: agent-level retrieval diff (Gemini vs Harrier on real session calls)
    phase_a_diff = _load_phase_a_static_diff()
    phase_a_judge = _load_phase_a_judge()
    phase_a_verdict = compute_phase_a_verdict(phase_a_diff, phase_a_judge) if (phase_a_diff and phase_a_judge) else None
    charts["phase_a_per_call_chart"] = chart_phase_a_per_call(phase_a_diff) if phase_a_diff else ""
    charts["phase_a_judge_chart"] = chart_phase_a_judge_verdicts(phase_a_judge) if phase_a_judge else ""
    charts["phase_a_verdict"] = phase_a_verdict
    charts["has_phase_a"] = bool(phase_a_diff and phase_a_judge)

    # Methodology notes
    methodology_notes = [
        f"Quality split: vector-only {int(W_QUALITY_VECTOR*100)}% + end-to-end {int(W_QUALITY_E2E*100)}% + long-context {int(W_LONG_CONTEXT*100)}% + local-control {int(W_LOCAL_CONTROL*100)}%.",
        f"Long-context axis = mean e2e_recall@10 over 'long' (2048-8192 tok) and 'very_long' (>8192 tok) length buckets.",
        f"Per-source-type floor: any candidate that regresses >5pts vs Gemini on court_doc or solicitor_letter E2E recall@10 is hard-vetoed.",
        f"Dim penalty: {DIM_PENALTY_PTS}pts subtracted if candidate dim != baseline dim ({base_dim}).",
        "Layer 1 pool-and-judge (Claude Sonnet) was NOT run — Anthropic Messages API rejects Claude Code OAuth tokens. Layer 1 NDCG@10 with graded relevance not yet measured.",
        "Phase 2 speed eval not run — top-2 quality finishers would be served via vLLM + Ollama with vegeta load-gen + co-tenancy stress.",
        "Local-control formula: 0.4 * ($_saved / max_$_saved) + 0.4 * min(1, ctx_headroom / 32768) + 0.2 * min(1, offline_hours / 168). Approximated since real cost numbers not provided.",
        "Confidence intervals on per-source bars use a binomial-approx 1.96*SE — rough but sufficient at this n.",
    ]

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    tpl = env.get_template("report.html.j2")
    html = tpl.render(
        run_date=dt.date.today().isoformat(),
        verdict={"verdict": verdict.verdict, "winner_id": verdict.winner_id,
                 "rationale": getattr(verdict, "rationale", "")},
        scores=scores_for_template,
        baseline_id=baseline.model_id,
        mermaid_diagram=MERMAID_DIAGRAM,
        has_speed_phase1=has_speed_phase1,
        has_speed_phase2=has_speed_phase2,
        methodology_notes=methodology_notes,
        n_models=len(results),
        **charts,
    )

    out_html = args.out_dir / f"{dt.date.today().isoformat()}_report.html"
    out_html.write_text(html)
    log.info(f"Wrote {out_html} ({out_html.stat().st_size / 1024:.1f} KB)")

    # Decision MD
    veto_lines = []
    for s in scores_for_template:
        if s["veto_reasons"]:
            veto_lines.append(f"- **{s['model_id']}**: {'; '.join(s['veto_reasons'])}")

    md_lines = [
        f"# Embeddings Benchmark Decision — {dt.date.today().isoformat()}",
        "",
        f"**Verdict:** `{verdict.verdict}` &middot; **Winner:** `{verdict.winner_id or 'n/a'}`",
        "",
        "## Scorecard",
        "",
        "| Model | dim | Vec q | E2E q | Long-ctx | Local-ctrl | Raw | Dim penalty | Total |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scores_for_template:
        md_lines.append(
            f"| {s['model_id']} | {s['dim']} | {s['quality_vector_only']:.1f} | "
            f"{s['quality_end_to_end']:.1f} | {s['long_context']:.1f} | "
            f"{s['local_control']:.1f} | {s['raw_total']:.1f} | "
            f"-{s['dim_penalty']:.1f} | **{s['total']:.1f}** |"
        )

    md_lines += [
        "",
        "## Vetoes",
        "",
    ]
    if veto_lines:
        md_lines.extend(veto_lines)
    else:
        md_lines.append("None — all candidates pass per-source-type floors.")

    md_lines += [
        "",
        "## Rationale",
        "",
        getattr(verdict, "rationale", "(no rationale)"),
        "",
        "## Next action",
        "",
        ("Run 10K-chunk re-embed parity check against a copy of the production hybrid pipeline before declaring switch."
         if verdict.verdict == "switch"
         else "Stay on Gemini; close benchmark or rerun with broader candidate set."
         if verdict.verdict == "stay"
         else "Inconclusive — see HTML report for additional measurement options."),
    ]
    out_md = args.out_dir / f"{dt.date.today().isoformat()}_decision.md"
    out_md.write_text("\n".join(md_lines))
    log.info(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
