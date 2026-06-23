"""Benchmark report writer.

Reads per-case measurements and emits ``results.json`` matching the Pattern C
design doc schema. If a ``baseline.json`` sits next to the output, regression
deltas vs the baseline are computed and embedded.

Programmatic use:
    from report import Report
    r = Report(topology="split_4090_llm_3090ti_audio")
    r.add_case("T02_medium_factual", ttft_audio_ms=691, wer=0.0, tokens=12)
    r.write(Path("results.json"))

CLI:
    python -m report --in cases.json --out results.json --baseline baseline.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from metrics import p50, p95


# Regression bands (ms) applied to TTFTAudio deltas
WARN_DELTA_MS = 50.0
FAIL_DELTA_MS = 150.0


@dataclass
class CaseResult:
    id: str
    ttft_audio_ms: float | None = None
    wer: float | None = None
    tokens: int | None = None
    turn_duration_ms: float | None = None
    stage_breakdown_ms: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    topology: str = "split_4090_llm_3090ti_audio"
    run_id: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    cases: list[CaseResult] = field(default_factory=list)
    baseline_id: str | None = None

    # --------------------------------------------------------------- mutation
    def add_case(self, case_id: str, **fields: Any) -> CaseResult:
        known = {f for f in CaseResult.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in fields.items() if k in known and k != "id"}
        extra = {k: v for k, v in fields.items() if k not in known}
        case = CaseResult(id=case_id, extra=extra, **kwargs)
        self.cases.append(case)
        return case

    # --------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        ttfts = [c.ttft_audio_ms for c in self.cases if c.ttft_audio_ms is not None]
        out: dict[str, Any] = {
            "n_cases": len(self.cases),
            "n_with_ttft": len(ttfts),
        }
        if ttfts:
            out["p50_ttft_ms"] = round(p50(ttfts), 2)
            out["p95_ttft_ms"] = round(p95(ttfts), 2)
            out["min_ttft_ms"] = round(min(ttfts), 2)
            out["max_ttft_ms"] = round(max(ttfts), 2)
        concurrencies = [
            c.extra.get("concurrency") for c in self.cases if c.extra.get("concurrency")
        ]
        if concurrencies:
            out["max_concurrent"] = max(concurrencies)
        return out

    # --------------------------------------------------------------- regression
    def regression(self, baseline: dict[str, Any] | None) -> dict[str, str]:
        if not baseline:
            return {}
        base_cases = {c["id"]: c for c in baseline.get("cases", [])}
        deltas: dict[str, str] = {}
        for case in self.cases:
            if case.ttft_audio_ms is None:
                continue
            base = base_cases.get(case.id)
            if not base or base.get("ttft_audio_ms") is None:
                continue
            delta = case.ttft_audio_ms - float(base["ttft_audio_ms"])
            band = "OK"
            if delta >= FAIL_DELTA_MS:
                band = "FAIL"
            elif delta >= WARN_DELTA_MS:
                band = "WARN"
            elif delta <= -WARN_DELTA_MS:
                band = "IMPROVED"
            sign = "+" if delta >= 0 else ""
            deltas[case.id] = f"{sign}{delta:.0f}ms vs baseline - {band}"
        return deltas

    # --------------------------------------------------------------- serialization
    def to_dict(self, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "baseline_id": baseline.get("run_id") if baseline else None,
            "topology": self.topology,
            "cases": [self._case_dict(c) for c in self.cases],
            "summary": self.summary(),
            "regression": self.regression(baseline),
        }

    @staticmethod
    def _case_dict(c: CaseResult) -> dict[str, Any]:
        d = asdict(c)
        extra = d.pop("extra", {}) or {}
        d = {k: v for k, v in d.items() if v not in (None, {}, [])}
        d.update(extra)
        return d

    def write(self, path: Path, baseline_path: Path | None = None) -> Path:
        baseline = None
        if baseline_path is None:
            baseline_path = path.with_name("baseline.json")
        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text())
            except Exception:
                baseline = None
        path.write_text(json.dumps(self.to_dict(baseline), indent=2, sort_keys=False))
        return path


# --------------------------------------------------------------------- CLI

def _from_input_file(in_path: Path) -> Report:
    payload = json.loads(in_path.read_text())
    rpt = Report(
        topology=payload.get("topology", "split_4090_llm_3090ti_audio"),
        run_id=payload.get("run_id", _dt.datetime.now().isoformat(timespec="seconds")),
    )
    for case in payload.get("cases", []):
        cid = case.pop("id")
        rpt.add_case(cid, **case)
    return rpt


def main() -> None:
    ap = argparse.ArgumentParser(description="write voice benchmark results.json")
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--baseline", dest="baseline_path", type=Path, default=None)
    args = ap.parse_args()

    rpt = _from_input_file(args.in_path)
    rpt.write(args.out_path, args.baseline_path)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
