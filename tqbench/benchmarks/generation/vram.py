"""GPU memory sampling — nvidia-smi for raw VRAM, /metrics for KV cache usage."""
from __future__ import annotations
import logging
import subprocess
import threading
import time

import httpx


class VRAMSampler:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mark_idx: int = 0

    def start(self) -> None:
        self._stop.clear()
        self.samples.clear()
        self._mark_idx = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def mark(self) -> int:
        """Return peak VRAM (MB) since the last mark() or start(), then reset the window."""
        window = self.samples[self._mark_idx:]
        self._mark_idx = len(self.samples)
        return max(window) if window else 0

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return max(self.samples) if self.samples else 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                mb = int(r.stdout.strip().splitlines()[0])
                self.samples.append(mb)
            except (ValueError, IndexError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval_s)


class ServerMetricsSampler:
    """Sample KV cache usage and request counts from an OpenAI-compatible server's /metrics endpoint."""

    def __init__(self, base_url: str, interval_s: float = 1.0):
        self.base_url = base_url
        self.interval_s = interval_s
        self.kv_usage_samples: list[float] = []
        self.running_requests_samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mark_idx: int = 0

    def start(self) -> None:
        self._stop.clear()
        self.kv_usage_samples.clear()
        self.running_requests_samples.clear()
        self._mark_idx = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def mark(self) -> dict:
        """Return peak KV cache usage since last mark(), then reset window."""
        kv_window = self.kv_usage_samples[self._mark_idx:]
        req_window = self.running_requests_samples[self._mark_idx:]
        self._mark_idx = len(self.kv_usage_samples)
        return {
            "kv_cache_usage_peak_pct": max(kv_window) * 100 if kv_window else 0,
            "kv_cache_usage_mean_pct": sum(kv_window) / len(kv_window) * 100 if kv_window else 0,
            "running_requests_peak": max(req_window) if req_window else 0,
            "n_samples": len(kv_window),
        }

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return {
            "kv_cache_usage_peak_pct": max(self.kv_usage_samples) * 100 if self.kv_usage_samples else 0,
            "total_samples": len(self.kv_usage_samples),
        }

    def _run(self) -> None:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        client = httpx.Client(base_url=self.base_url, timeout=5)
        while not self._stop.is_set():
            try:
                r = client.get("/metrics")
                if r.status_code == 200:
                    self._parse_metrics(r.text)
            except Exception:
                pass
            self._stop.wait(self.interval_s)
        client.close()

    def _parse_metrics(self, text: str) -> None:
        for line in text.splitlines():
            if line.startswith("vllm:kv_cache_usage_perc{"):
                val = float(line.split("} ")[1])
                self.kv_usage_samples.append(val)
            elif line.startswith("vllm:num_requests_running{"):
                val = int(float(line.split("} ")[1]))
                self.running_requests_samples.append(val)
