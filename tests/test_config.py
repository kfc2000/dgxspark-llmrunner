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
