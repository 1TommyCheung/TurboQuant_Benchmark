"""Metrics computation for the voice benchmark.

All timings are in milliseconds and derive from monotonic event logs produced
by ``harness.VoiceSession``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

try:
    from jiwer import wer as _jiwer_wer
except ImportError:  # pragma: no cover
    _jiwer_wer = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- core

def event_time(events: Iterable[dict[str, Any]], name: str) -> float | None:
    for ev in events:
        if ev.get("event") == name:
            return float(ev["t"])
    return None


def ttft_audio_ms(events: Sequence[dict[str, Any]]) -> float:
    """First audio sample out relative to last audio chunk sent in."""
    t_last_sent = event_time(events, "audio_chunk_sent_last")
    t_first_audio = event_time(events, "tts_first_audio_sample")
    if t_last_sent is None or t_first_audio is None:
        raise ValueError("missing audio_chunk_sent_last or tts_first_audio_sample event")
    return (t_first_audio - t_last_sent) * 1000.0


def turn_duration_ms(events: Sequence[dict[str, Any]]) -> float:
    """Full turn: first audio sent in -> last audio sample out."""
    t_first_sent = event_time(events, "audio_chunk_sent_first")
    t_last_audio = event_time(events, "tts_last_audio_sample")
    if t_last_audio is None:
        last = None
        for ev in events:
            if ev.get("event") in ("tts_audio_chunk", "tts_first_audio_sample"):
                last = float(ev["t"])
        t_last_audio = last
    if t_first_sent is None or t_last_audio is None:
        raise ValueError("missing audio_chunk_sent_first or tts audio events")
    return (t_last_audio - t_first_sent) * 1000.0


def stage_breakdown_ms(events: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per-stage latencies. Returns only stages whose anchors are present."""
    anchors = {
        "audio_in": "audio_chunk_sent_last",
        "vad_end": "vad_end_detected",
        "stt": "stt_result_received",
        "llm_first_token": "llm_first_token",
        "tts_first_audio": "tts_first_audio_sample",
    }
    times = {k: event_time(events, v) for k, v in anchors.items()}
    order = ["audio_in", "vad_end", "stt", "llm_first_token", "tts_first_audio"]
    out: dict[str, float] = {}
    prev = times["audio_in"]
    for name in order[1:]:
        cur = times[name]
        if prev is not None and cur is not None:
            out[name] = (cur - prev) * 1000.0
            prev = cur
        elif cur is not None:
            prev = cur
    return out


# --------------------------------------------------------------------------- WER

def compute_wer(reference: str, hypothesis: str) -> float:
    """Word error rate between reference and hypothesis transcripts."""
    if _jiwer_wer is None:
        raise RuntimeError("jiwer not installed; cannot compute WER")
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return float(_jiwer_wer(reference, hypothesis))


# --------------------------------------------------------------------------- aggregates

def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be in [0, 100]")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def p50(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def p95(values: Sequence[float]) -> float:
    return percentile(values, 95.0)


def summarize(measurements: Sequence[float]) -> dict[str, float]:
    if not measurements:
        return {"count": 0}
    return {
        "count": len(measurements),
        "min": min(measurements),
        "max": max(measurements),
        "mean": sum(measurements) / len(measurements),
        "p50": p50(measurements),
        "p95": p95(measurements),
    }
