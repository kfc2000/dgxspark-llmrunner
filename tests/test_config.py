import json

import pytest

from llmrunner import config


def test_load_valid_config(tmp_path):
    p = tmp_path / "llms.json"
    p.write_text(
        json.dumps(
            {
                "llms": [
                    {
                        "name": "a",
                        "type": "docker_vllm",
                        "workdir": "/tmp",
                        "start_script": "./s.sh",
                        "stop_script": "./t.sh",
                        "endpoint": "http://x:8000/",
                        "container": "c1",
                    }
                ]
            }
        )
    )
    llms = config.load_config(p)
    assert len(llms) == 1
    assert llms[0].endpoint == "http://x:8000"
    assert llms[0].container == "c1"


def test_missing_keys_raise(tmp_path):
    p = tmp_path / "llms.json"
    p.write_text('{"llms": [{"name": "a"}]}')
    with pytest.raises(config.ConfigError, match="missing keys"):
        config.load_config(p)


def test_bad_type_raises(tmp_path):
    p = tmp_path / "llms.json"
    p.write_text(
        json.dumps(
            {
                "llms": [
                    {
                        "name": "a",
                        "type": "podman",
                        "workdir": "/tmp",
                        "start_script": "s",
                        "stop_script": "t",
                    }
                ]
            }
        )
    )
    with pytest.raises(config.ConfigError, match="unknown type"):
        config.load_config(p)


def test_missing_file_returns_empty(tmp_path):
    assert config.load_config(tmp_path / "nope.json") == []


def test_sparkrun_types_accepted(tmp_path):
    p = tmp_path / "llms.json"
    p.write_text(
        json.dumps(
            {
                "llms": [
                    {
                        "name": "a",
                        "type": "sparkrun_vllm",
                        "workdir": "/tmp",
                        "start_script": "s",
                        "stop_script": "t",
                    },
                    {
                        "name": "b",
                        "type": "sparkrun_sglang",
                        "workdir": "/tmp",
                        "start_script": "s",
                        "stop_script": "t",
                    },
                ]
            }
        )
    )
    llms = config.load_config(p)
    assert [llm.type for llm in llms] == ["sparkrun_vllm", "sparkrun_sglang"]


def test_is_sparkrun_type() -> None:
    assert config.is_sparkrun_type("sparkrun_vllm")
    assert config.is_sparkrun_type("sparkrun_sglang")
    assert not config.is_sparkrun_type("docker_vllm")
    assert not config.is_sparkrun_type("docker_sglang")
