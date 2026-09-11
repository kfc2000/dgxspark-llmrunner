from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from llmrunner import hw_metrics

MAX_POINTS = 300


@dataclass
class HwSample:
    ts: float
    sys: dict = field(default_factory=dict)
    gpu: hw_metrics.GpuStat | None = None


class HwSampler:
    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.history: deque[HwSample] = deque(maxlen=MAX_POINTS)
        self.latest: HwSample | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        import psutil

        psutil.cpu_percent()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import psutil

        while not self._stop.is_set():
            sample = HwSample(ts=time.time(), sys=hw_metrics.read_system())
            sample.gpu = hw_metrics.read_gpu()
            with self._lock:
                self.history.append(sample)
                self.latest = sample
            self._stop.wait(self.interval_s)
            psutil.cpu_percent()

    def snapshot(self) -> tuple[HwSample | None, list[HwSample]]:
        with self._lock:
            return self.latest, list(self.history)


_sampler: HwSampler | None = None


def get_sampler() -> HwSampler:
    global _sampler
    if _sampler is None:
        _sampler = HwSampler()
        _sampler.start()
    return _sampler
