from llmrunner.config import LLMConfig
from llmrunner.control import START_TIMEOUT_S, container_logs, is_llm_running, running_llms, sparkrun_logs, start_llm


class _FakeResult:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _patch_run(monkeypatch, result: _FakeResult) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        calls.append(args)
        return result

    monkeypatch.setattr("llmrunner.control.subprocess.run", fake_run)
    return calls


def test_container_logs_tail(monkeypatch) -> None:
    calls = _patch_run(monkeypatch, _FakeResult(stdout="a\nb\n"))
    out = container_logs("sglang-x", 300)
    assert calls == [["docker", "logs", "--tail", "300", "sglang-x"]]
    assert out == "a\nb\n"


def test_container_logs_all(monkeypatch) -> None:
    calls = _patch_run(monkeypatch, _FakeResult(stdout="raw\r\nline"))
    out = container_logs("sglang-x", None)
    assert calls == [["docker", "logs", "sglang-x"]]
    assert out == "raw\r\nline"


def test_sparkrun_logs_tail(monkeypatch) -> None:
    calls = _patch_run(monkeypatch, _FakeResult(stdout="a\nb\n"))
    out = sparkrun_logs("qwen3.8-27b", 200)
    assert calls == [["sparkrun", "logs", "qwen3.8-27b", "-n", "200"]]
    assert out == "a\nb\n"


def test_sparkrun_logs_all(monkeypatch) -> None:
    calls = _patch_run(monkeypatch, _FakeResult(stdout="raw"))
    out = sparkrun_logs("qwen3.8-27b", None)
    assert calls == [["sparkrun", "logs", "qwen3.8-27b"]]
    assert out == "raw"


def test_start_timeout_is_30_minutes() -> None:
    assert START_TIMEOUT_S == 30 * 60


def _llm(name: str) -> LLMConfig:
    return LLMConfig(
        name=name,
        type="docker_vllm",
        workdir="/tmp",
        start_script=f"./start_{name}.sh",
        stop_script=f"./stop_{name}.sh",
    )


def test_start_llm_stops_others_first(monkeypatch) -> None:
    seen: list[str] = []

    def fake_run_script(script, workdir, timeout=None):  # noqa: ANN001
        seen.append(script)
        return (0, "")

    monkeypatch.setattr("llmrunner.control.run_script", fake_run_script)
    ok, output = start_llm(_llm("b"), stop_first=[_llm("a")])
    assert ok
    assert output == "started"
    assert seen == ["./stop_a.sh", "./start_b.sh"]


def test_start_llm_aborts_when_stop_fails(monkeypatch) -> None:
    seen: list[str] = []

    def fake_run_script(script, workdir, timeout=None):  # noqa: ANN001
        seen.append(script)
        return (1, "stop failed") if "stop" in script else (0, "")

    monkeypatch.setattr("llmrunner.control.run_script", fake_run_script)
    ok, output = start_llm(_llm("b"), stop_first=[_llm("a")])
    assert not ok
    assert "could not stop" in output and "a" in output
    assert seen == ["./stop_a.sh"]


def test_running_llms_filters_by_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "llmrunner.control.is_llm_running",
        lambda llm: (llm.name == "a", "up"),
    )
    out = running_llms([_llm("a"), _llm("b")])
    assert [llm.name for llm in out] == ["a"]


def test_is_llm_running_sparkrun_uses_endpoint(monkeypatch) -> None:
    llm = LLMConfig(
        name="qwen3.8-27b",
        type="sparkrun_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        endpoint="http://localhost:8000",
        container="qwen3.8-27b",
    )
    monkeypatch.setattr("llmrunner.control.endpoint_alive", lambda ep: True)
    running, detail = is_llm_running(llm)
    assert running and "endpoint responding" in detail
