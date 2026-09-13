from __future__ import annotations

import math
import os
from pathlib import Path

import psutil
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles

from llmrunner import config, control, hw_metrics, llm_metrics
from llmrunner.sampler import get_sampler

WINDOW_S = 300
REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"

app = FastAPI(title="DGX Spark LLM Runner API")
sampler = get_sampler()
tracker = llm_metrics.get_tracker()


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _require_token(authorization: str | None = Header(None)) -> None:
    token = os.environ.get("LLMRUNNER_TOKEN")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="missing or bad bearer token")


def _load_llms() -> list[config.LLMConfig]:
    try:
        return config.load_config()
    except config.ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _find(name: str) -> config.LLMConfig:
    for llm in _load_llms():
        if llm.name == name:
            return llm
    raise HTTPException(status_code=404, detail=f"unknown model: {name}")


def _pair_series(history, key: str, gpu_attr: str | None = None) -> list[list]:
    out = []
    for h in history:
        v = getattr(h.gpu, gpu_attr, None) if (gpu_attr and h.gpu) else (None if gpu_attr else h.sys.get(key))
        out.append([h.ts, _clean(v)])
    return out


@app.get("/api/hw")
def api_hw() -> dict:
    latest, history = sampler.snapshot()
    if latest is None:
        return {"ready": False}
    freq = psutil.cpu_freq()
    gpu = latest.gpu
    series = {
        key: _pair_series(history, key)
        for key in ("cpu_pct", "cpu_freq_mhz", "cpu_temp_c", "mem_pct",
                    "disk_pct", "disk_read_mbps", "disk_write_mbps")
    }
    gpu_series = {}
    if gpu is not None:
        gpu_series = {
            key: _pair_series(history, None, key)
            for key in ("util_pct", "clock_mhz", "temp_c", "power_w")
        }
    return {
        "ready": True,
        "ts": latest.ts,
        "window_s": WINDOW_S,
        "sys": {k: _clean(v) for k, v in latest.sys.items()},
        "cpu_freq_max": freq.max or None,
        "gpu": None if gpu is None else {
            "name": gpu.name,
            "util_pct": gpu.util_pct,
            "clock_mhz": gpu.clock_mhz,
            "temp_c": gpu.temp_c,
            "power_w": gpu.power_w,
            "power_limit_w": gpu.power_limit_w,
        },
        "series": series,
        "gpu_series": gpu_series,
        "nvml_error": None if hw_metrics.gpu_available() else (hw_metrics._nvml.error or None),
    }


@app.get("/api/llms")
def api_llms() -> dict:
    out = []
    for llm in _load_llms():
        running, detail = control.is_llm_running(llm)
        rates = tracker.scrape(llm.name, llm.endpoint) if running else None
        reachable = bool(rates and rates.reachable)
        out.append({
            "name": llm.name,
            "type": llm.type,
            "endpoint": llm.endpoint,
            "container": llm.container,
            "running": running,
            "detail": detail,
            "reachable": reachable,
            "decode_tps": rates.decode_tps if reachable else None,
            "prefill_tps": rates.prefill_tps if reachable else None,
            "running_requests": rates.running_requests if reachable else None,
            "history": [{"ts": p["ts"], "decode_tps": p["decode_tps"],
                         "prefill_tps": p["prefill_tps"]} for p in rates.history] if reachable else [],
        })
    return {"llms": out}


@app.post("/api/llms/{name}/start")
def api_start(name: str, _: None = Depends(_require_token)) -> dict:
    llm = _find(name)
    others = [o for o in _load_llms() if o.name != name and control.is_llm_running(o)[0]]
    try:
        ok, output = control.start_llm(llm, stop_first=others)
    except control.ControlError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": ok, "output": output[-4000:]}


@app.post("/api/llms/{name}/stop")
def api_stop(name: str, _: None = Depends(_require_token)) -> dict:
    llm = _find(name)
    try:
        ok, output = control.stop_llm(llm)
    except control.ControlError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": ok, "output": output[-4000:]}


TAIL_OPTIONS = (10, 20, 50, 100)


@app.get("/api/logs")
def api_logs(name: str, tail: int = 20) -> dict:
    llm = _find(name)
    if not llm.container:
        raise HTTPException(status_code=400, detail=f"{name}: no 'container' set in llms.json")
    if tail not in TAIL_OPTIONS:
        raise HTTPException(status_code=400, detail=f"invalid 'tail'={tail}; must be one of {', '.join(map(str, TAIL_OPTIONS))}")
    try:
        if config.is_sparkrun_type(llm.type):
            logs = control.sparkrun_logs(llm.container, tail)
        else:
            logs = control.container_logs(llm.container, tail)
    except control.ControlError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"logs": logs, "lines": len(logs.splitlines())}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
