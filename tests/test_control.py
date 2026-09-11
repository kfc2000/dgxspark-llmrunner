from llmrunner.control import _normalize_log_stream


def test_carriage_returns_become_newlines():
    raw = "a\r\nb\rc\rd"
    out = _normalize_log_stream(raw)
    assert "\r" not in out
    assert out.splitlines() == ["a", "b", "c", "d"]
