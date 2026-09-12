from llmrunner.llm_metrics import parse_prometheus

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
