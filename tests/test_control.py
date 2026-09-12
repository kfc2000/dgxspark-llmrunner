from llmrunner.control import START_TIMEOUT_S, container_logs


class _FakeResult:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _patch_docker(monkeypatch, result: _FakeResult) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        calls.append(args)
        return result

    monkeypatch.setattr("llmrunner.control.subprocess.run", fake_run)
    return calls


def test_container_logs_tail(monkeypatch) -> None:
    calls = _patch_docker(monkeypatch, _FakeResult(stdout="a\nb\n"))
    out = container_logs("sglang-x", 300)
    assert calls == [["docker", "logs", "--tail", "300", "sglang-x"]]
    assert out == "a\nb\n"


def test_container_logs_all(monkeypatch) -> None:
    calls = _patch_docker(monkeypatch, _FakeResult(stdout="raw\r\nline"))
    out = container_logs("sglang-x", None)
    assert calls == [["docker", "logs", "sglang-x"]]
    assert out == "raw\r\nline"


def test_start_timeout_is_30_minutes() -> None:
    assert START_TIMEOUT_S == 30 * 60
