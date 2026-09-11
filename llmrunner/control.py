from __future__ import annotations

import subprocess

from llmrunner.config import LLMConfig


class ControlError(Exception):
    pass


def _docker(args: list[str], timeout: float = 15.0) -> str:
    try:
        out = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ControlError("docker CLI not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ControlError(f"`docker {' '.join(args)}` timed out") from exc
    if out.returncode != 0:
        raise ControlError(out.stderr.strip() or f"docker {' '.join(args)} failed")
    return out.stdout


def container_status(name: str) -> str | None:
    try:
        out = _docker(["ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Status}}"])
    except ControlError:
        return None
    line = out.strip().splitlines()
    return line[0] if line else None


def container_logs(name: str, tail: int = 300) -> str:
    return _docker(["logs", "--tail", str(tail), name], timeout=30.0)


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


def is_llm_running(llm: LLMConfig) -> tuple[bool, str]:
    if llm.container:
        status = container_status(llm.container)
        if status is None:
            return False, "container not found"
        if status.lower().startswith("up"):
            return True, status
        return False, status
    alive = endpoint_alive(llm.endpoint)
    return alive, "endpoint responding" if alive else "endpoint unreachable"


def run_script(script: str, workdir: str, timeout: float = 120.0) -> tuple[int, str]:
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


def start_llm(llm: LLMConfig) -> tuple[bool, str]:
    rc, output = run_script(llm.start_script, llm.workdir)
    return rc == 0, output or (f"exit {rc}" if rc else "started")


def stop_llm(llm: LLMConfig) -> tuple[bool, str]:
    rc, output = run_script(llm.stop_script, llm.workdir)
    return rc == 0, output or (f"exit {rc}" if rc else "stopped")
