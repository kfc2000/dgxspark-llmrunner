from __future__ import annotations

import math
import time
from datetime import datetime

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import altair as alt
import pandas as pd
import streamlit as st

from llmrunner import config, llm_metrics
from llmrunner.sampler import get_sampler

st.set_page_config(page_title="DGX Spark LLM Runner", page_icon="▦", layout="wide")

llms = config.load_config()
sampler = get_sampler()
tracker = llm_metrics.get_tracker()


def _fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    if value is None or isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}{unit}"


WINDOW_S = 300


def _window_x():
    now = time.time()
    return alt.X(
        "t:T",
        title=None,
        axis=alt.Axis(format="%H:%M:%S"),
        scale=alt.Scale(domain=[datetime.fromtimestamp(now - WINDOW_S), datetime.fromtimestamp(now)]),
    )


def _sparkline(values: list[tuple[float, float]]) -> None:
    now = time.time()
    recent = [(datetime.fromtimestamp(ts), v) for ts, v in values if now - ts <= WINDOW_S]
    df = pd.DataFrame(recent, columns=["t", "v"])
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x=_window_x(), y=alt.Y("v:Q", title=None, scale=alt.Scale(domain=[0, 100])))
    )
    st.altair_chart(chart, height=140, use_container_width=True)


@st.fragment(run_every="2s")
def render_hardware() -> None:
    latest, history = sampler.snapshot()
    if latest is None:
        st.info("Waiting for the first sample…")
        return
    s = latest.sys
    gpu = latest.gpu

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CPU", _fmt(s.get("cpu_pct"), " %"), _fmt(s.get("cpu_temp_c"), " °C"), delta_color="inverse")
        st.caption(f"load {s.get('load_1m', 0):.2f} · {_fmt(s.get('cpu_freq_mhz') or None, ' MHz', 0)}")
    with c2:
        st.metric(
            "CPU power",
            _fmt(s.get("cpu_power_w"), " W"),
            _fmt(s.get("cpu_temp_c"), " °C"),
            delta_color="inverse",
        )
    with c3:
        mem_delta = f"{s.get('mem_used_gb', 0):.1f} / {s.get('mem_total_gb', 0):.0f} GiB"
        st.metric("Memory", _fmt(s.get("mem_pct"), " %"), mem_delta)
    with c4:
        disk_delta = f"{s.get('disk_used_gb', 0):.0f} / {s.get('disk_total_gb', 0):.0f} GiB"
        st.metric("Storage /", _fmt(s.get("disk_pct"), " %"), disk_delta)

    st.divider()
    if gpu is not None:
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("GPU util", _fmt(gpu.util_pct, " %"))
        g2.metric("GPU temp", _fmt(gpu.temp_c, " °C"), delta_color="inverse")
        g3.metric("GPU power", _fmt(gpu.power_w, " W"), f"limit {_fmt(gpu.power_limit_w, ' W')}")
        g4.metric(
            "VRAM",
            _fmt(None if gpu.mem_used_mb is None else gpu.mem_used_mb / 1024, " GiB", 1),
            f"of {(gpu.mem_total_mb or 0) / 1024:.0f} GiB",
        )
    else:
        from llmrunner.hw_metrics import _nvml

        st.warning(
            "GPU metrics unavailable via NVML"
            + (f" ({_nvml.error})" if _nvml.error else "")
            + ". Check the NVIDIA driver, or expose sensors through "
            "/sys/class/hwmon (read by the CPU temperature/power probes)."
        )

    st.divider()
    t1, t2, t3 = st.columns(3)
    with t1:
        st.caption("CPU %")
        _sparkline([(h.ts, h.sys.get("cpu_pct", 0.0)) for h in history])
    with t2:
        st.caption("Memory %")
        _sparkline([(h.ts, h.sys.get("mem_pct", 0.0)) for h in history])
    with t3:
        st.caption("GPU util")
        _sparkline([(h.ts, (h.gpu.util_pct or 0.0) if h.gpu else 0.0) for h in history])


@st.fragment(run_every="3s")
def render_llms() -> None:
    if not llms:
        st.info(f"No LLMs configured. Create `{config.DEFAULT_CONFIG_PATH}` (see repo `llms.json`).")
        return
    for group_start in range(0, len(llms), 4):
        group = llms[group_start : group_start + 4]
        cols = st.columns(len(group))
        for col, llm in zip(cols, group, strict=True):
            with col:
                rates = tracker.scrape(llm.name, llm.endpoint)
                st.caption(f"**{llm.name}** · {llm.type}")
                if not rates.reachable:
                    st.metric("status", "offline", "no /metrics")
                    continue
                st.metric("Decode", _fmt(rates.decode_tps, " tok/s"))
                st.metric("Prefill", _fmt(rates.prefill_tps, " tok/s"))
                now = time.time()
                recent = [p for p in rates.history if now - p["ts"] <= WINDOW_S]
                rows = (
                    [
                        {"t": datetime.fromtimestamp(p["ts"]), "tok/s": p["decode_tps"], "kind": "decode"}
                        for p in recent
                    ]
                    + [
                        {"t": datetime.fromtimestamp(p["ts"]), "tok/s": p["prefill_tps"], "kind": "prefill"}
                        for p in recent
                    ]
                )
                if rows:
                    df = pd.DataFrame(rows)
                    chart = (
                        alt.Chart(df)
                        .mark_line()
                        .encode(x=_window_x(), y=alt.Y("tok/s:Q", title=None), color=alt.Color("kind:N"))
                    )
                    st.altair_chart(chart, height=160, use_container_width=True)


st.title("DGX Spark · LLM Runner")

render_hardware()
st.divider()
st.subheader("LLM throughput")
render_llms()
