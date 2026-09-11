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
