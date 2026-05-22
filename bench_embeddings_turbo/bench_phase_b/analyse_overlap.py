"""Cross-backend chunk-overlap analysis across Phase B+ complex scenarios.

For each scenario, compute pairwise Jaccard@cited and Jaccard@retrieved between
V1, V2, and Qwen3-FP8. Highlights where backends diverge most.
"""
from __future__ import annotations
import json
from pathlib import Path
from itertools import combinations


OUT_DIR = Path("/mnt/i/dev/Legal/case_kb/bench_embeddings/reports/raw/phase_b_plus")
SCENARIOS = ["scenario_1_multi_hop", "scenario_2_timeline", "scenario_3_privilege", "scenario_4_correction"]
BACKENDS = ["v1", "v2", "qwen"]


def load(scenario: str, backend: str) -> dict:
    p = OUT_DIR / f"{scenario}__{backend}.json"
    return json.loads(p.read_text())


def cited_set(data: dict) -> set:
    out = set()
    for t in data.get("turns", []):
        out.update(t.get("cited_chunk_ids", []))
    return out


def retrieved_set(data: dict) -> set:
    out = set()
    for t in data.get("turns", []):
        for s in t.get("searches", []):
            out.update(s.get("top_chunk_ids", []))
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    report_lines = []
    report_lines.append("# Phase B+ Cross-Backend Overlap Analysis\n")
    report_lines.append(f"Generated from {OUT_DIR}\n")

    overall = {}
    for scenario in SCENARIOS:
        report_lines.append(f"\n## {scenario}\n")
        loaded = {b: load(scenario, b) for b in BACKENDS}
        cited = {b: cited_set(loaded[b]) for b in BACKENDS}
        retrieved = {b: retrieved_set(loaded[b]) for b in BACKENDS}

        report_lines.append("| Backend | Cited unique | Retrieved unique | Searches |")
        report_lines.append("|---|---:|---:|---:|")
        for b in BACKENDS:
            n_searches = sum(len(t.get("searches", [])) for t in loaded[b].get("turns", []))
            report_lines.append(f"| {b} | {len(cited[b])} | {len(retrieved[b])} | {n_searches} |")

        report_lines.append("\n**Pairwise Jaccard (cited):**")
        for a, b in combinations(BACKENDS, 2):
            j = jaccard(cited[a], cited[b])
            report_lines.append(f"- {a} ↔ {b}: **{j:.3f}** ({len(cited[a] & cited[b])} common / {len(cited[a] | cited[b])} total)")

        report_lines.append("\n**Pairwise Jaccard (retrieved):**")
        for a, b in combinations(BACKENDS, 2):
            j = jaccard(retrieved[a], retrieved[b])
            report_lines.append(f"- {a} ↔ {b}: **{j:.3f}** ({len(retrieved[a] & retrieved[b])} common / {len(retrieved[a] | retrieved[b])} total)")

        # Find chunks that V1 cited but Qwen didn't retrieve at all
        v1_cited_only = cited["v1"] - retrieved["v2"] - retrieved["qwen"]
        v2_cited_only = cited["v2"] - retrieved["v1"] - retrieved["qwen"]
        qwen_cited_only = cited["qwen"] - retrieved["v1"] - retrieved["v2"]
        report_lines.append("\n**Backend-exclusive citations (chunks cited by one, never retrieved by others):**")
        report_lines.append(f"- v1-only: {len(v1_cited_only)} chunks: {sorted(v1_cited_only)[:5]}{'...' if len(v1_cited_only) > 5 else ''}")
        report_lines.append(f"- v2-only: {len(v2_cited_only)} chunks: {sorted(v2_cited_only)[:5]}{'...' if len(v2_cited_only) > 5 else ''}")
        report_lines.append(f"- qwen-only: {len(qwen_cited_only)} chunks: {sorted(qwen_cited_only)[:5]}{'...' if len(qwen_cited_only) > 5 else ''}")

        overall[scenario] = {
            "v1_v2_cited_jaccard": jaccard(cited["v1"], cited["v2"]),
            "v1_qwen_cited_jaccard": jaccard(cited["v1"], cited["qwen"]),
            "v2_qwen_cited_jaccard": jaccard(cited["v2"], cited["qwen"]),
            "v1_v2_retrieved_jaccard": jaccard(retrieved["v1"], retrieved["v2"]),
            "v1_qwen_retrieved_jaccard": jaccard(retrieved["v1"], retrieved["qwen"]),
            "v2_qwen_retrieved_jaccard": jaccard(retrieved["v2"], retrieved["qwen"]),
        }

    report_lines.append("\n\n## Overall summary table (cited Jaccard)\n")
    report_lines.append("| Scenario | V1↔V2 | V1↔Qwen | V2↔Qwen | Avg |")
    report_lines.append("|---|---:|---:|---:|---:|")
    for scenario, d in overall.items():
        avg = (d["v1_v2_cited_jaccard"] + d["v1_qwen_cited_jaccard"] + d["v2_qwen_cited_jaccard"]) / 3
        report_lines.append(f"| {scenario.replace('scenario_', '').replace('_', ' ')} | {d['v1_v2_cited_jaccard']:.3f} | {d['v1_qwen_cited_jaccard']:.3f} | {d['v2_qwen_cited_jaccard']:.3f} | {avg:.3f} |")

    text = "\n".join(report_lines)
    out_path = OUT_DIR / "_overlap_report.md"
    out_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
