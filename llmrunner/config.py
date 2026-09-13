from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

VALID_TYPES = {"docker_vllm", "docker_sglang", "sparkrun_vllm", "sparkrun_sglang"}
SPARKRUN_TYPES = {"sparkrun_vllm", "sparkrun_sglang"}


def is_sparkrun_type(llm_type: str) -> bool:
    return llm_type in SPARKRUN_TYPES

DEFAULT_CONFIG_PATH = os.environ.get("LLMRUNNER_CONFIG", "llms.json")


class ConfigError(Exception):
    pass


@dataclass
class LLMConfig:
    name: str
    type: str
    workdir: str
    start_script: str
    stop_script: str
    endpoint: str = "http://localhost:8000"
    container: str | None = None
    raw: dict = field(default_factory=dict, repr=False)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> list[LLMConfig]:
    path = Path(path)
    if not path.is_absolute():
        candidates = [path, _REPO_ROOT / path]
        path = next((c for c in candidates if c.exists()), candidates[0])
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from exc

    entries = data.get("llms") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ConfigError(f"{path}: expected a list under 'llms'")

    llms: list[LLMConfig] = []
    for i, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: entry {i} is not an object")
        missing = [k for k in ("name", "type", "workdir", "start_script", "stop_script") if k not in item]
        if missing:
            raise ConfigError(f"{path}: entry {i} missing keys: {', '.join(missing)}")
        if item["type"] not in VALID_TYPES:
            raise ConfigError(
                f"{path}: entry {i} has unknown type {item['type']!r}, expected one of {sorted(VALID_TYPES)}"
            )
        llms.append(
            LLMConfig(
                name=item["name"],
                type=item["type"],
                workdir=item["workdir"],
                start_script=item["start_script"],
                stop_script=item["stop_script"],
                endpoint=str(item.get("endpoint", "http://localhost:8000")).rstrip("/"),
                container=item.get("container"),
                raw=item,
            )
        )
    return llms
