from __future__ import annotations

import subprocess

from llmrunner.config import LLMConfig, get_sparkrun_path, is_sparkrun_type


class ControlError(Exception):
    pass


def _run_cmd(cmd: list[str], timeout: float, merge_stderr: bool = False) -> str:
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ControlError(f"{cmd[0]} CLI not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ControlError(f"`{' '.join(cmd)}` timed out") from exc
    if out.returncode != 0:
        raise ControlError((out.stderr or out.stdout).strip() or f"{' '.join(cmd)} failed")
    return out.stdout


def _docker(args: list[str], timeout: float = 15.0, merge_stderr: bool = False) -> str:
    return _run_cmd(["docker", *args], timeout, merge_stderr)


def _sparkrun(args: list[str], timeout: float = 15.0, merge_stderr: bool = False) -> str:
    return _run_cmd([get_sparkrun_path(), *args], timeout, merge_stderr)


def container_status(target: str) -> str | None:
    """Return the docker status line for the container matching ``target`` by
    name or id (including an abbreviated id), or None if it is not present."""
    try:
        out = _docker(["ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}"])
    except ControlError:
        return None
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        cid, cname, status = parts
        if target in (cid, cname) or cid.startswith(target):
            return status
    return None


def container_logs(name: str, tail: int | None = 300) -> str:
    args = ["logs"]
    if tail is not None:
        args += ["--tail", str(tail)]
    args.append(name)
    return _docker(args, timeout=30.0, merge_stderr=True)


def sparkrun_logs(target: str, tail: int | None = 300) -> str:
    args = ["logs", target]
    if tail is not None:
        args += ["-n", str(tail)]
    return _sparkrun(args, timeout=30.0, merge_stderr=True)


def sparkrun_status(timeout: float = 15.0) -> str:
    return _sparkrun(["status"], timeout, merge_stderr=True)


def sparkrun_process_running(sparkrun_id: str) -> bool:
    try:
        out = sparkrun_status()
    except ControlError:
        return False
    return any(sparkrun_id in line for line in out.splitlines())


def running_containers() -> list[str]:
    try:
        out = _docker(["ps", "--format", "{{.Names}}"])
    except ControlError:
        return []
    return [line for line in out.splitlines() if line]


def endpoint_alive(endpoint: str, timeout: float = 2.0) -> bool:
    import requests

    for path in ("/health", "/metrics", "/v1/models"):
        try:
            r = requests.get(endpoint.rstrip("/") + path, timeout=timeout)
            if r.status_code < 500:
                return True
        except Exception:
            continue
    return False


def models_served(endpoint: str, timeout: float = 2.0) -> list[str]:
    import requests

    try:
        r = requests.get(endpoint.rstrip("/") + "/v1/models", timeout=timeout)
        if r.status_code >= 500:
            return []
        data = r.json()
    except Exception:
        return []
    return [str(item.get("id")) for item in data.get("data", []) if item.get("id")]


def model_available(endpoint: str, model_name: str, timeout: float = 2.0) -> bool:
    return model_name in models_served(endpoint, timeout)


def container_active(name: str) -> bool:
    status = container_status(name)
    if status is None:
        return False
    s = status.lower()
    return s.startswith("up") or s.startswith("restarting")


def llm_status(llm: LLMConfig) -> tuple[str, str]:
    """Return (state, detail) where state is 'running', 'starting', or 'stopped'.

    A model is 'running' when its name is served at the /v1/models endpoint.
    It is 'starting' when its container or sparkrun process is up, but the model
    is not yet served (a 404, an empty /v1/models response, or an error).
    Otherwise it is 'stopped'.
    """
    if model_available(llm.endpoint, llm.name):
        return "running", "model served at /v1/models"
    if is_sparkrun_type(llm.type):
        if llm.sparkrun_id and sparkrun_process_running(llm.sparkrun_id):
            return "starting", "sparkrun process running"
        if llm.container_id and container_active(llm.container_id):
            return "starting", container_status(llm.container_id) or "container present"
    elif llm.container_id and container_active(llm.container_id):
        return "starting", container_status(llm.container_id) or "container present"
    return "stopped", "not running"


def is_llm_running(llm: LLMConfig) -> tuple[bool, str]:
    state, detail = llm_status(llm)
    return state != "stopped", detail


START_TIMEOUT_S = 30 * 60


def run_script(script: str, workdir: str, timeout: float | None = 120.0) -> tuple[int, str]:
    import os

    if not os.path.isdir(workdir):
        raise ControlError(f"workdir does not exist: {workdir}")
    try:
        proc = subprocess.run(
            ["bash", "-c", script],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ControlError("bash not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise ControlError(f"script timed out after {timeout}s") from exc
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode, output


def stop_llm(llm: LLMConfig) -> tuple[bool, str]:
    rc, output = run_script(llm.stop_script, llm.workdir)
    return rc == 0, output or (f"exit {rc}" if rc else "stopped")


def start_llm(llm: LLMConfig, stop_first: list[LLMConfig] | None = None) -> tuple[bool, str]:
    for other in stop_first or []:
        ok, output = stop_llm(other)
        if not ok:
            return False, f"aborted: could not stop running model {other.name!r}: {output}"
    rc, output = run_script(llm.start_script, llm.workdir, timeout=START_TIMEOUT_S)
    return rc == 0, output or (f"exit {rc}" if rc else "started")


def running_llms(llms: list[LLMConfig]) -> list[LLMConfig]:
    return [llm for llm in llms if is_llm_running(llm)[0]]
