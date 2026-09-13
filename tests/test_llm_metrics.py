from llmrunner.llm_metrics import parse_prometheus, sum_by_label

SAMPLE = """\
# HELP vllm:prompt_tokens_total Total number of prompt tokens processed
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="qwen3"} 4000.0
vllm:prompt_tokens_total{model_name="qwen3"} 1000.0
vllm:generation_tokens_total{model_name="qwen3"} 2500.0
vllm:request_prefill_time_seconds_sum{model_name="qwen3"} 8.0
vllm:num_requests_running 3
broken line without value
"""


def test_parser_sums_labelled_series():
    totals = parse_prometheus(SAMPLE)
    assert totals["vllm:prompt_tokens_total"] == 5000.0
    assert totals["vllm:generation_tokens_total"] == 2500.0
    assert totals["vllm:request_prefill_time_seconds_sum"] == 8.0
    assert totals["vllm:num_requests_running"] == 3.0


def test_parser_ignores_comments_and_garbage():
    totals = parse_prometheus("# comment\n\nnot_a_metric\n")
    assert totals == {}


def test_parser_value_with_spaces_in_labels():
    totals = parse_prometheus('metric{label="a b"} 7.5')
    assert totals["metric"] == 7.5


def test_sum_by_label_groups_by_mode():
    text = '\n'.join([
        'sglang:realtime_tokens_total{mode="decode"} 100.0',
        'sglang:realtime_tokens_total{mode="prefill_compute"} 200.0',
        'sglang:realtime_tokens_total{model="q",mode="prefill_cache"} 30.0',
    ])
    grouped = sum_by_label(text, "sglang:realtime_tokens_total", "mode")
    assert grouped == {"decode": 100.0, "prefill_compute": 200.0, "prefill_cache": 30.0}


def test_sum_by_label_ignores_other_metrics():
    text = 'sglang:gen_throughput 12.0\nsglang:realtime_tokens_total{mode="decode"} 5.0'
    assert sum_by_label(text, "sglang:realtime_tokens_total", "mode") == {"decode": 5.0}


def test_two_scrapes_compute_rates_without_keyerror(monkeypatch):
    from llmrunner.llm_metrics import LlmMetricTracker

    class _Clock:
        def __init__(self):
            self.value = 1000.0

        def __call__(self):
            return self.value

    clock = _Clock()
    monkeypatch.setattr("llmrunner.llm_metrics.time.time", clock)

    def make_resp(**counters):
        text = "\n".join(f"{k} {v}" for k, v in counters.items())

        class _Resp:
            raise_for_status = lambda: None  # noqa: E731

        _Resp.text = text
        return _Resp

    first_totals = {
        "vllm:prompt_tokens_total": 4000.0,
        "vllm:generation_tokens_total": 2500.0,
        "vllm:request_prefill_time_seconds_sum": 8.0,
    }

    def fake_get(url, timeout):
        return make_resp(**first_totals)

    tracker = LlmMetricTracker()
    monkeypatch.setattr("llmrunner.llm_metrics.requests.get", fake_get)

    tracker.scrape("m", "http://x")  # seeds counters

    clock.value += 2.0
    first_totals["vllm:generation_tokens_total"] += 30.0
    first_totals["vllm:prompt_tokens_total"] += 40.0
    first_totals["vllm:request_prefill_time_seconds_sum"] += 2.0

    second = tracker.scrape("m", "http://x")

    assert second.reachable
    assert not second.error
    assert second.decode_tps == 15.0  # 30 tokens / 2s
    assert second.prefill_tps == 20.0  # 40 tokens / 2s prefill time


def test_sglang_uses_live_gen_throughput_gauge(monkeypatch):
    from llmrunner.llm_metrics import LlmMetricTracker

    class _Clock:
        def __init__(self):
            self.value = 1000.0

        def __call__(self):
            return self.value

    clock = _Clock()
    monkeypatch.setattr("llmrunner.llm_metrics.time.time", clock)

    def make_resp(**counters):
        text = "\n".join(f"{k} {v}" for k, v in counters.items())

        class _Resp:
            raise_for_status = lambda: None  # noqa: E731

        _Resp.text = text
        return _Resp

    counters = {
        "sglang:gen_throughput": 10.0,
        "sglang:prompt_tokens_total": 4000.0,
        "sglang:generation_tokens_total": 2500.0,
        "sglang:prefill_effective_tokens_total": 100.0,
        "sglang:num_running_reqs": 2,
    }

    def fake_get(url, timeout):
        return make_resp(**counters)

    tracker = LlmMetricTracker()
    monkeypatch.setattr("llmrunner.llm_metrics.requests.get", fake_get)

    # gen_throughput is a live gauge: even the first scrape reports decode tps.
    first = tracker.scrape("m", "http://x")
    assert first.decode_tps == 10.0

    clock.value += 2.0
    counters["sglang:gen_throughput"] = 25.0
    counters["sglang:prefill_effective_tokens_total"] = 180.0

    second = tracker.scrape("m", "http://x")

    assert second.reachable
    assert second.decode_tps == 25.0  # gauge read directly
    assert second.prefill_tps == 40.0  # (180 - 100) / 2s
    assert second.running_requests == 2.0  # sglang:num_running_reqs


def test_sglang_realtime_tokens_delta_is_preferred(monkeypatch):
    from llmrunner.llm_metrics import LlmMetricTracker

    class _Clock:
        def __init__(self):
            self.value = 1000.0

        def __call__(self):
            return self.value

    clock = _Clock()
    monkeypatch.setattr("llmrunner.llm_metrics.time.time", clock)

    def make_resp(**counters):
        text = "\n".join(f"{k} {v}" for k, v in counters.items())

        class _Resp:
            raise_for_status = lambda: None  # noqa: E731

        _Resp.text = text
        return _Resp

    counters = {
        'sglang:realtime_tokens_total{mode="decode"}': 100.0,
        'sglang:realtime_tokens_total{mode="prefill_compute"}': 200.0,
        'sglang:realtime_tokens_total{mode="prefill_cache"}': 30.0,
        "sglang:gen_throughput": 12.0,
    }

    def fake_get(url, timeout):
        return make_resp(**counters)

    tracker = LlmMetricTracker()
    monkeypatch.setattr("llmrunner.llm_metrics.requests.get", fake_get)

    tracker.scrape("m", "http://x")  # seeds counters

    clock.value += 2.0
    counters['sglang:realtime_tokens_total{mode="decode"}'] = 160.0
    counters['sglang:realtime_tokens_total{mode="prefill_compute"}'] = 240.0
    # prefill_cache unchanged at 30.0
    counters["sglang:gen_throughput"] = 12.0  # gauge unchanged

    second = tracker.scrape("m", "http://x")

    assert second.decode_tps == 30.0  # (160 - 100) / 2s, not the stale gauge
    assert second.prefill_tps == 20.0  # ((240-200)+(30-30)) / 2s


def test_background_tick_collects_and_latest_returns_cached(monkeypatch):
    from llmrunner.llm_metrics import LlmMetricTracker

    class _LLM:
        def __init__(self, name, endpoint):
            self.name = name
            self.endpoint = endpoint

    class _Clock:
        def __init__(self):
            self.value = 1000.0

        def __call__(self):
            return self.value

    clock = _Clock()
    monkeypatch.setattr("llmrunner.llm_metrics.time.time", clock)

    def make_resp(text):
        class _Resp:
            raise_for_status = lambda: None  # noqa: E731

        _Resp.text = text
        return _Resp

    def fake_get(url, timeout):
        return make_resp("sglang:gen_throughput 7.5\n")

    tracker = LlmMetricTracker()
    monkeypatch.setattr("llmrunner.llm_metrics.requests.get", fake_get)
    tracker.start_collecting(
        lambda: [_LLM("a", "http://a"), _LLM("b", "http://b")], interval_s=1.0
    )

    tracker._tick()

    assert tracker.latest("a").reachable
    assert tracker.latest("a").decode_tps == 7.5
    assert tracker.latest("b").reachable
    assert tracker.latest("nonexistent") is None

    tracker._stop.set()
