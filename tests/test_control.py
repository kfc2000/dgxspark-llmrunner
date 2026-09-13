from llmrunner.config import LLMConfig
from llmrunner.control import (
    START_TIMEOUT_S,
    container_logs,
    is_llm_running,
    llm_status,
    model_available,
    models_served,
    running_llms,
    sparkrun_logs,
    sparkrun_process_running,
    start_llm,
)


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


def test_sparkrun_logs_uses_configured_path(monkeypatch) -> None:
    calls = _patch_run(monkeypatch, _FakeResult(stdout="a\nb\n"))
    monkeypatch.setattr("llmrunner.control.get_sparkrun_path", lambda: "/opt/bin/sparkrun")
    out = sparkrun_logs("qwen3.8-27b", 200)
    assert calls == [["/opt/bin/sparkrun", "logs", "qwen3.8-27b", "-n", "200"]]
    assert out == "a\nb\n"


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
        container_id="qwen3.8-27b",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: True)
    running, detail = is_llm_running(llm)
    assert running and "model served" in detail


def test_llm_status_running_when_model_served(monkeypatch) -> None:
    llm = LLMConfig(
        name="a",
        type="docker_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="c1",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: True)
    state, detail = llm_status(llm)
    assert state == "running"
    assert "model served" in detail


def test_llm_status_not_running_when_model_not_served(monkeypatch) -> None:
    llm = LLMConfig(
        name="a",
        type="docker_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="c1",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: False)
    monkeypatch.setattr("llmrunner.control.container_status", lambda name: None)
    monkeypatch.setattr("llmrunner.control.endpoint_alive", lambda ep: False)
    state, detail = llm_status(llm)
    assert state == "stopped"


def test_llm_status_docker_starting_when_container_present(monkeypatch) -> None:
    llm = LLMConfig(
        name="a",
        type="docker_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="c1",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: False)
    monkeypatch.setattr("llmrunner.control.container_status", lambda name: "Up 5 seconds")
    state, detail = llm_status(llm)
    assert state == "starting"
    assert "Up 5 seconds" in detail


def test_llm_status_docker_stopped_when_container_absent(monkeypatch) -> None:
    llm = LLMConfig(
        name="a",
        type="docker_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="c1",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: False)
    monkeypatch.setattr("llmrunner.control.container_status", lambda name: None)
    monkeypatch.setattr("llmrunner.control.endpoint_alive", lambda ep: False)
    state, detail = llm_status(llm)
    assert state == "stopped"
    assert "not running" in detail


def test_llm_status_docker_starting_when_endpoint_responds(monkeypatch) -> None:
    llm = LLMConfig(
        name="a",
        type="docker_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="c1",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: False)
    monkeypatch.setattr("llmrunner.control.container_status", lambda name: None)
    monkeypatch.setattr("llmrunner.control.endpoint_alive", lambda ep: True)
    state, detail = llm_status(llm)
    assert state == "starting"
    assert "model not served" in detail


def test_llm_status_docker_stopped_when_container_exited(monkeypatch) -> None:
    llm = LLMConfig(
        name="a",
        type="docker_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="c1",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: False)
    monkeypatch.setattr("llmrunner.control.container_status", lambda name: "Exited (0) 1 minute ago")
    monkeypatch.setattr("llmrunner.control.endpoint_alive", lambda ep: False)
    state, detail = llm_status(llm)
    assert state == "stopped"


def test_llm_status_sparkrun_starting_when_process_present(monkeypatch) -> None:
    llm = LLMConfig(
        name="qwen3.8-27b",
        type="sparkrun_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="qwen3.8-27b",
        sparkrun_id="qwen3.8-27b",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: False)
    monkeypatch.setattr("llmrunner.control.sparkrun_process_running", lambda sid: True)
    state, detail = llm_status(llm)
    assert state == "starting"
    assert "sparkrun process running" in detail


def test_llm_status_sparkrun_stopped_when_process_absent(monkeypatch) -> None:
    llm = LLMConfig(
        name="qwen3.8-27b",
        type="sparkrun_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="qwen3.8-27b",
        sparkrun_id="qwen3.8-27b",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: False)
    monkeypatch.setattr("llmrunner.control.sparkrun_process_running", lambda sid: False)
    monkeypatch.setattr("llmrunner.control.container_status", lambda name: None)
    monkeypatch.setattr("llmrunner.control.endpoint_alive", lambda ep: False)
    state, detail = llm_status(llm)
    assert state == "stopped"


def test_is_llm_running_true_for_starting(monkeypatch) -> None:
    llm = LLMConfig(
        name="a",
        type="docker_vllm",
        workdir="/tmp",
        start_script="./start.sh",
        stop_script="./stop.sh",
        container_id="c1",
    )
    monkeypatch.setattr("llmrunner.control.model_available", lambda ep, name: False)
    monkeypatch.setattr("llmrunner.control.container_status", lambda name: "Up 5 seconds")
    running, detail = is_llm_running(llm)
    assert running is True


def test_models_served_parses_ids(monkeypatch) -> None:
    import requests as _requests

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "qwen3-32b"}, {"id": "gpt-oss-20b"}]}

    monkeypatch.setattr(_requests, "get", lambda url, timeout=2.0: _Resp())
    assert models_served("http://x:8000") == ["qwen3-32b", "gpt-oss-20b"]


def test_model_available_true(monkeypatch) -> None:
    monkeypatch.setattr(
        "llmrunner.control.models_served",
        lambda endpoint, timeout=2.0: ["qwen3-32b"],
    )
    assert model_available("http://x:8000", "qwen3-32b") is True


def test_model_available_false(monkeypatch) -> None:
    monkeypatch.setattr(
        "llmrunner.control.models_served",
        lambda endpoint, timeout=2.0: ["qwen3-32b"],
    )
    assert model_available("http://x:8000", "other") is False


def test_sparkrun_process_running_true(monkeypatch) -> None:
    monkeypatch.setattr(
        "llmrunner.control.sparkrun_status",
        lambda timeout=15.0: "NAME\nqwen3.8-27b\nother\n",
    )
    assert sparkrun_process_running("qwen3.8-27b") is True


def test_sparkrun_process_running_false_when_absent(monkeypatch) -> None:
    monkeypatch.setattr(
        "llmrunner.control.sparkrun_status",
        lambda timeout=15.0: "NAME\nother\n",
    )
    assert sparkrun_process_running("qwen3.8-27b") is False


def test_sparkrun_process_running_false_on_error(monkeypatch) -> None:
    from llmrunner.control import ControlError

    def boom(timeout=15.0):
        raise ControlError("sparkrun not found")

    monkeypatch.setattr("llmrunner.control.sparkrun_status", boom)
    assert sparkrun_process_running("qwen3.8-27b") is False
