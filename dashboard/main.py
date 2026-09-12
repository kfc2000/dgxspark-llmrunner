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


def _sparkline(
    values: list[tuple[float, float]], y_domain: tuple[float, float] | None = (0, 100)
) -> None:
    now = time.time()
    recent = [
        (datetime.fromtimestamp(ts), v)
        for ts, v in values
        if now - ts <= WINDOW_S and v is not None and not math.isnan(v)
    ]
    df = pd.DataFrame(recent, columns=["t", "v"])
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x=_window_x(), y=alt.Y("v:Q", title=None, scale=alt.Scale(domain=list(y_domain)) if y_domain else None))
    )
    st.altair_chart(chart, height=110, width="stretch")


def _series(history, key: str, gpu_attr: str | None = None) -> list[tuple[float, float]]:
    out = []
    for h in history:
        if gpu_attr is not None:
            v = getattr(h.gpu, gpu_attr, None) if h.gpu else None
        else:
            v = h.sys.get(key)
        out.append((h.ts, v))
    return out


@st.fragment(run_every="2s")
def render_hardware() -> None:
    latest, history = sampler.snapshot()
    if latest is None:
        st.info("Waiting for the first sample…")
        return
    s = latest.sys
    gpu = latest.gpu

    sys_cards = [
        (
            "CPU Util",
            _fmt(s.get("cpu_pct"), " %"),
            f"load {s.get('load_1m', 0):.2f}",
            _series(history, "cpu_pct"),
            (0, 100),
        ),
        (
            "CPU freq",
            _fmt(s.get("cpu_freq_mhz") or None, " MHz", 0),
            "",
            _series(history, "cpu_freq_mhz"),
            (0, 4800),
        ),
        (
            "CPU temp",
            _fmt(s.get("cpu_temp_c"), " °C"),
            "",
            _series(history, "cpu_temp_c"),
            (0, 100),
        ),
        (
            "Memory",
            _fmt(s.get("mem_pct"), " %"),
            f"{s.get('mem_used_gb', 0):.1f} / {s.get('mem_total_gb', 0):.0f} GiB",
            _series(history, "mem_pct"),
            (0, 100),
        ),
    ]

    if gpu is None:
        gpu_cards = []
    else:
        gpu_cards = [
            (
                "GPU Util",
                _fmt(gpu.util_pct, " %"),
                "",
                _series(history, None, "util_pct"),
                (0, 100),
            ),
            (
                "GPU Freq",
                _fmt(getattr(gpu, "clock_mhz", None), " MHz", 0),
                "",
                _series(history, None, "clock_mhz"),
                (0, 3003),
            ),
            (
                "GPU Temp",
                _fmt(gpu.temp_c, " °C"),
                "",
                _series(history, None, "temp_c"),
                (0, 100),
            ),
            (
                "GPU Power",
                _fmt(gpu.power_w, " W"),
                f"limit {_fmt(gpu.power_limit_w, ' W')}",
                _series(history, None, "power_w"),
                (0, 100),
            ),
        ]

    storage_cards = [
        (
            "Storage /",
            _fmt(s.get("disk_pct"), " %"),
            f"{s.get('disk_used_gb', 0):.0f} / {s.get('disk_total_gb', 0):.0f} GiB",
            _series(history, "disk_pct"),
            (0, 100),
        ),
        (
            "Disk read",
            _fmt(s.get("disk_read_mbps"), " MB/s"),
            "",
            _series(history, "disk_read_mbps"),
            _auto_domain(_series(history, "disk_read_mbps")),
        ),
        (
            "Disk write",
            _fmt(s.get("disk_write_mbps"), " MB/s"),
            "",
            _series(history, "disk_write_mbps"),
            _auto_domain(_series(history, "disk_write_mbps")),
        ),
    ]

    _card_row(sys_cards)
    st.divider()
    if gpu_cards:
        _card_row(gpu_cards)
    else:
        from llmrunner.hw_metrics import _nvml

        st.warning(
            "GPU metrics unavailable via NVML"
            + (f" ({_nvml.error})" if _nvml.error else "")
            + ". Check the NVIDIA driver, or expose sensors through "
            "/sys/class/hwmon (read by the CPU temperature/power probes)."
        )
    st.divider()
    _card_row(storage_cards, width=4)


def _auto_domain(series: list[tuple[float, float]]) -> tuple[float, float]:
    vals = [v for _, v in series if v is not None and not math.isnan(v)]
    top = max(max(vals, default=50.0) * 1.2, 50.0)
    return (0, float(int((top + 49) // 50 * 50)))


def _card_row(cards, width: int = 4) -> None:
    cols = st.columns(width)
    for col, card in zip(cols, cards, strict=False):
        label, value, sub, series, domain = card
        with col:
            st.metric(label, value)
            if sub:
                st.caption(sub)
            _sparkline(series, domain)
    for col in cols[len(cards):]:
        with col:
            st.write("")


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
                    st.altair_chart(chart, height=160, width="stretch")


st.title("DGX Spark · LLM Runner")

render_hardware()
st.divider()
st.subheader("LLM throughput")
render_llms()
