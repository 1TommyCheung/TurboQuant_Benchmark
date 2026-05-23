"""VRAM sampling via nvidia-smi in a background thread."""
from __future__ import annotations
import subprocess
import threading
import time


class VRAMSampler:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

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
