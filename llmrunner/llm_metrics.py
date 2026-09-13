from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import requests

PROMPT_METRICS = (
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens",
    "sglang:prompt_tokens_total",
)
GEN_METRICS = (
    "vllm:generation_tokens_total",
    "vllm:generation_tokens",
    "sglang:generation_tokens_total",
)
RUNNING_METRICS = (
    "vllm:num_requests_running",
    "sglang:num_running_reqs",
    "sglang:num_running_requests",
)
PREFILL_TIME_METRICS = (
    "vllm:request_prefill_time_seconds_sum",
    "sglang:prefill_time_seconds_sum",
)
# SGLang bumps prompt/generation_tokens_total only when a request finishes, so
# deltas of those counters read 0 while a stream is in flight. Prefer the live
# gauge / log-interval counters that move during generation.
GEN_THROUGHPUT_GAUGES = ("sglang:gen_throughput",)
PREFILL_EFFECTIVE_METRICS = ("sglang:prefill_effective_tokens_total",)


def parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1) if " " in line else []
        if len(parts) != 2:
            continue
        ident, raw_value = parts
        name = ident.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def _lookup(totals: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in totals:
            return totals[name]
    return None


@dataclass
class LlmRates:
    name: str
    endpoint: str
    reachable: bool = False
    decode_tps: float | None = None
    prefill_tps: float | None = None
    running_requests: float | None = None
    total_generated: float | None = None
    error: str = ""
    history: deque = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = deque(maxlen=MAX_POINTS)


MAX_POINTS = 300


class LlmMetricTracker:
    def __init__(self, timeout_s: float = 3.0) -> None:
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._counters: dict[str, tuple[float, dict[str, float]]] = {}
        self._rates: dict[str, LlmRates] = {}
        self._provider = None
        self._interval_s = 1.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def latest(self, name: str) -> LlmRates | None:
        with self._lock:
            return self._rates.get(name)

    def start_collecting(self, provider, interval_s: float = 1.0) -> None:
        """Scrape every configured LLM's /metrics in a background thread.

        ``provider`` is a zero-arg callable returning a list of objects with
        ``.name`` and ``.endpoint`` (e.g. ``config.load_config``). This keeps the
        token-throughput history fresh at ``interval_s`` regardless of how often
        the API is polled; status checks stay on their own cadence.
        """
        self._provider = provider
        self._interval_s = interval_s
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self._interval_s)

    def _tick(self) -> None:
        if self._provider is None:
            return
        try:
            llms = self._provider()
        except Exception:
            return
        for llm in llms:
            name = getattr(llm, "name", None)
            endpoint = getattr(llm, "endpoint", None)
            if name is None or endpoint is None:
                continue
            try:
                self.scrape(name, endpoint)
            except Exception:
                continue

    def scrape(self, name: str, endpoint: str) -> LlmRates:
        url = endpoint.rstrip("/") + "/metrics"
        try:
            resp = requests.get(url, timeout=self.timeout_s)
            resp.raise_for_status()
            totals = parse_prometheus(resp.text)
        except Exception as exc:
            with self._lock:
                rates = self._rates.get(name) or LlmRates(name=name, endpoint=endpoint)
                rates.reachable = False
                rates.error = str(exc)
                self._rates[name] = rates
                return rates

        prompt = _lookup(totals, PROMPT_METRICS)
        generated = _lookup(totals, GEN_METRICS)
        prefill_s = _lookup(totals, PREFILL_TIME_METRICS)
        running = _lookup(totals, RUNNING_METRICS)
        gen_tps = _lookup(totals, GEN_THROUGHPUT_GAUGES)
        prefill_eff = _lookup(totals, PREFILL_EFFECTIVE_METRICS)
        now = time.time()
        counters = {
            "prompt": prompt,
            "generated": generated,
            "prefill_s": prefill_s,
            "gen_tps": gen_tps,
            "prefill_eff": prefill_eff,
            "now": now,
        }

        decode_tps = prefill_tps = None
        with self._lock:
            prev = self._counters.get(name)
            self._counters[name] = (now, counters)
            rates = self._rates.get(name) or LlmRates(name=name, endpoint=endpoint)
            rates.endpoint = endpoint
            dt = counters["now"] - prev[1]["now"] if prev is not None else 0.0
            # Decode tok/s: SGLang exports a live gen_throughput gauge (tokens/s)
            # that moves while a stream is in flight. Fall back to the delta of
            # the cumulative generation-token counter for vLLM.
            if gen_tps is not None:
                decode_tps = gen_tps
            elif prev is not None and dt > 0.5:
                if generated is not None and prev[1]["generated"] is not None:
                    d_gen = generated - prev[1]["generated"]
                    if d_gen >= 0:
                        decode_tps = d_gen / dt
            # Prefill tok/s: prefer the per-log-interval counter delta, then the
            # prefill-time-based rate, then prompt-token delta over wall time.
            if prev is not None and dt > 0.5:
                if prefill_eff is not None and prev[1]["prefill_eff"] is not None:
                    d_pref = prefill_eff - prev[1]["prefill_eff"]
                    if d_pref >= 0:
                        prefill_tps = d_pref / dt
                elif (
                    prompt is not None
                    and prefill_s is not None
                    and prev[1]["prompt"] is not None
                    and prev[1]["prefill_s"] is not None
                ):
                    d_prompt = prompt - prev[1]["prompt"]
                    d_time = prefill_s - prev[1]["prefill_s"]
                    if d_prompt >= 0 and d_time > 0.05:
                        prefill_tps = d_prompt / d_time
                elif prompt is not None and prev[1]["prompt"] is not None:
                    d_prompt = prompt - prev[1]["prompt"]
                    if d_prompt >= 0:
                        prefill_tps = d_prompt / dt
            rates.reachable = True
            rates.error = ""
            rates.running_requests = running
            rates.total_generated = generated
            if decode_tps is not None or prefill_tps is not None:
                rates.decode_tps = decode_tps if decode_tps is not None else 0.0
                rates.prefill_tps = prefill_tps if prefill_tps is not None else 0.0
                rates.history.append(
                    {
                        "ts": now,
                        "decode_tps": rates.decode_tps,
                        "prefill_tps": rates.prefill_tps,
                    }
                )
            self._rates[name] = rates
            return rates


_tracker: LlmMetricTracker | None = None


def get_tracker() -> LlmMetricTracker:
    global _tracker
    if _tracker is None:
        _tracker = LlmMetricTracker()
    return _tracker
