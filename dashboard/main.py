from __future__ import annotations

import math
import time

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import streamlit as st
import theme

from llmrunner import config, control, llm_metrics
from llmrunner.config import ConfigError
from llmrunner.sampler import get_sampler

st.set_page_config(page_title="DGX Spark LLM Runner", page_icon="▦", layout="wide")

try:
    llms = config.load_config()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

sampler = get_sampler()
tracker = llm_metrics.get_tracker()

WINDOW_S = 300
SPARK_W = 200
SPARK_H = 38


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _last(values: list[tuple[float, float]], count: int = 60) -> list[float]:
    now = time.time()
    vals = [
        v for ts, v in values
        if now - ts <= WINDOW_S and v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if len(vals) > count:
        stride = math.ceil(len(vals) / count)
        vals = vals[::stride]
    return vals


def _spark_svg(values: list[tuple[float, float]], color: str, domain: tuple[float, float] | None = None) -> str:
    pts = _last(values)
    svg_open = f'<svg class="lr-spark" viewBox="0 0 {SPARK_W} {SPARK_H}" preserveAspectRatio="none">'
    if len(pts) < 2:
        return svg_open + "</svg>"
    if domain is not None:
        lo, hi = domain
    else:
        lo = min(0.0, min(pts))
        hi = max(pts) * 1.2 or 1.0
    if hi <= lo:
        hi = lo + 1.0
    step = SPARK_W / (len(pts) - 1)
    d = " ".join(
        f"{'M' if i == 0 else 'L'}{i * step:.1f},"
        f"{SPARK_H - 2 - (min(max(v, lo), hi) - lo) / (hi - lo) * (SPARK_H - 4):.1f}"
        for i, v in enumerate(pts)
    )
    return svg_open + f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/></svg>'


def _card(label: str, value: str, unit: str, sub: str, series, color: str, domain=None) -> str:
    return (
        f'<section class="lr-card"><div class="lr-head">'
        f'<span class="lr-label">{_html_escape(label)}</span>'
        f'<span class="lr-sub">{_html_escape(sub)}</span></div>'
        f'<div class="lr-value">{value}<small>{unit}</small></div>'
        f"{_spark_svg(series, color, domain)}</section>"
    )


def _card_row(cards: list[str], three: bool = False) -> None:
    cls = "lr-row lr-row3" if three else "lr-row"
    st.markdown(f'<div class="{cls}">{"".join(cards)}</div>', unsafe_allow_html=True)


def _peak(values: list[tuple[float, float]]) -> float | None:
    pts = _last(values, 999)
    return max(pts) if pts else None


@st.fragment(run_every="2s")
def render_hardware() -> None:
    latest, history = sampler.snapshot()
    if latest is None:
        st.info("Waiting for the first sample…")
        return
    s = latest.sys
    gpu = latest.gpu

    import psutil

    freq_max = psutil.cpu_freq().max or 0.0

    cpu_temp_peak = _peak(_series(history, "cpu_temp_c"))
    sys_cards = [
        _card("CPU Util", _fmt(s.get("cpu_pct")), " %", f"load {s.get('load_1m', 0):.2f}",
              _series(history, "cpu_pct"), "var(--accent)", (0, 100)),
        _card("CPU Freq", _fmt((s.get("cpu_freq_mhz") or 0) / 1000, 2), " GHz", f"max {freq_max / 1000:.1f} GHz",
              [(ts, (v or 0) / 1000) for ts, v in _series(history, "cpu_freq_mhz")],
              "var(--accent)", (0, freq_max / 1000 or 4.8)),
        _card("CPU Temp", _fmt(s.get("cpu_temp_c"), 0), " °C",
              f"peak {cpu_temp_peak:.0f}°C" if cpu_temp_peak else "",
              _series(history, "cpu_temp_c"), "var(--warn)", (0, 100)),
        _card("Memory", _fmt(s.get("mem_pct"), 0), " %",
              f"{s.get('mem_used_gb', 0):.1f} / {s.get('mem_total_gb', 0):.0f} GiB",
              _series(history, "mem_pct"), "var(--accent)", (0, 100)),
    ]
    _card_row(sys_cards)

    if gpu is None:
        from llmrunner.hw_metrics import _nvml

        st.warning(
            "GPU metrics unavailable via NVML"
            + (f" ({_nvml.error})" if _nvml.error else "")
            + ". Check the NVIDIA driver, or expose sensors through /sys/class/hwmon."
        )
    else:
        gpu_temp_peak = _peak(_series(history, None, "temp_c"))
        gpu_freq_peak = _peak(_series(history, None, "clock_mhz"))
        gpu_cards = [
            _card("GPU Util", _fmt(gpu.util_pct, 0), " %", _html_escape(gpu.name or ""),
                  _series(history, None, "util_pct"), "var(--accent-2)", (0, 100)),
            _card("GPU Freq", _fmt(gpu.clock_mhz, 0), " MHz",
                  f"max {gpu_freq_peak:.0f} MHz" if gpu_freq_peak else "",
                  _series(history, None, "clock_mhz"), "var(--accent)", (0, max(gpu_freq_peak or 3003, 1))),
            _card("GPU Temp", _fmt(gpu.temp_c, 0), " °C",
                  f"peak {gpu_temp_peak:.0f}°C" if gpu_temp_peak else "",
                  _series(history, None, "temp_c"), "var(--warn)", (0, 100)),
            _card("GPU Power", _fmt(gpu.power_w, 0), " W",
                  f"limit {_fmt(gpu.power_limit_w, 0)} W" if gpu.power_limit_w else "",
                  _series(history, None, "power_w"), "var(--accent-2)", (0, max(gpu.power_limit_w or 100, 1))),
        ]
        _card_row(gpu_cards)

    storage_cards = [
        _card("Storage /", _fmt(s.get("disk_pct"), 0), " %",
              f"{s.get('disk_used_gb', 0):.0f} / {s.get('disk_total_gb', 0):.0f} GiB",
              _series(history, "disk_pct"), "var(--accent)", (0, 100)),
        _card("Disk Read", _fmt(s.get("disk_read_mbps")), " MB/s", "",
              _series(history, "disk_read_mbps"), "var(--accent)"),
        _card("Disk Write", _fmt(s.get("disk_write_mbps")), " MB/s", "",
              _series(history, "disk_write_mbps"), "var(--accent-2)"),
    ]
    _card_row(storage_cards, three=True)


def _series(history, key: str, gpu_attr: str | None = None) -> list[tuple[float, float]]:
    out = []
    for h in history:
        if gpu_attr is not None:
            v = getattr(h.gpu, gpu_attr, None) if h.gpu else None
        else:
            v = h.sys.get(key)
        out.append((h.ts, v))
    return out


def _hero_chart(history_points: list[dict]) -> str:
    now = time.time()
    pts = [p for p in history_points if now - p["ts"] <= WINDOW_S]
    defs = (
        '<defs><linearGradient id="gDec" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="var(--accent)" stop-opacity=".28"/>'
        '<stop offset="1" stop-color="var(--accent)" stop-opacity="0"/>'
        "</linearGradient></defs>"
    )
    grid = (
        '<g stroke="var(--border)" stroke-width="1">'
        '<line x1="0" y1="37" x2="700" y2="37"/>'
        '<line x1="0" y1="74" x2="700" y2="74"/>'
        '<line x1="0" y1="111" x2="700" y2="111"/></g>'
    )

    def path(key: str) -> tuple[str, float]:
        pairs = [(p["ts"], p[key]) for p in pts if p.get(key) is not None]
        if len(pairs) < 2:
            return "", 0.0
        hi = max(max(v for _, v in pairs) * 1.2, 1.0)
        t0 = now - WINDOW_S
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{max((ts - t0) / WINDOW_S * 700, 0):.0f},{150 - (v / hi) * 140:.0f}"
            for i, (ts, v) in enumerate(pairs)
        )
        return d, pairs[0][0]

    dec, dec_x0 = path("decode_tps")
    pre, _ = path("prefill_tps")
    parts = [defs, grid]
    if dec:
        x0 = max((dec_x0 - (now - WINDOW_S)) / WINDOW_S * 700, 0)
        parts.append(f'<path fill="url(#gDec)" d="{dec} L700,150 L{x0:.0f},150 Z"/>')
    if pre:
        parts.append(f'<path fill="none" stroke="var(--warn)" stroke-width="1.6" stroke-opacity=".85" d="{pre}"/>')
    if dec:
        parts.append(f'<path fill="none" stroke="var(--accent)" stroke-width="2.2" stroke-linejoin="round" d="{dec}"/>')
    return f'<div class="lr-chart"><svg viewBox="0 0 700 150" preserveAspectRatio="none">{"".join(parts)}</svg></div>'


def _throughput_col(label: str, current: float | None, history_points: list[dict], key: str) -> str:
    vals = [p[key] for p in history_points if p.get(key) is not None]
    avg = sum(vals) / len(vals) if vals else None
    if current is None or (isinstance(current, float) and math.isnan(current)):
        delta = ""
    elif avg is None:
        delta = ""
    else:
        d = current - avg
        cls = "up" if d >= 0 else "down"
        arrow = "▲" if d >= 0 else "▼"
        delta = f'<div class="lr-delta {cls}">{arrow} {abs(d):.1f} vs 5m avg</div>'
    return (
        f'<div class="lr-t"><div class="lr-label">{label}</div>'
        f'<div class="lr-value">{_fmt(current)}<small>tok/s</small></div>{delta}</div>'
    )


@st.fragment(run_every="3s")
def render_hero() -> None:
    hero = st.container(horizontal=True)
    names = [llm.name for llm in llms]

    side, prefill_col, decode_col, chart_col = hero.columns([5, 2, 2, 6])
    if not names:
        with side:
            st.info(f"No LLMs configured. Create `{config.DEFAULT_CONFIG_PATH}` (see repo `llms.json`).")
        return

    status_map = {llm.name: control.is_llm_running(llm) for llm in llms}
    selected = _selected_llm(names, status_map)
    running, detail = status_map[selected.name]

    with side:
        st.selectbox("Model", names, index=names.index(selected.name), key="model_sel",
                     label_visibility="collapsed")
        st.markdown(
            f'<div class="lr-sub">{_html_escape(selected.type)} · {_html_escape(selected.endpoint)}</div>',
            unsafe_allow_html=True,
        )
        reachable = False
        rates = None
        if running:
            rates = tracker.scrape(selected.name, selected.endpoint)
            reachable = rates.reachable
        if running and reachable:
            pill = ("running", "RUNNING")
        elif running:
            pill = ("starting", "STARTING")
        else:
            pill = ("stopped", "STOPPED")
        c1, c2, c3, c4 = st.columns([2.2, 1.2, 1, 1])
        with c1:
            st.markdown(f'<span class="lr-status {pill[0]}"><i></i> {pill[1]}</span>', unsafe_allow_html=True)
        others = [o for o in llms if status_map[o.name][0] and o.name != selected.name]
        do_start = c2.button("Swap & Start" if others else "Start", key="btn_start",
                             disabled=running and reachable, type="primary", width="stretch")
        do_stop = c3.button("Stop", key="btn_stop", disabled=not running, width="stretch")
        do_reload = c4.button("Reload", key="btn_reload", width="stretch")
        if do_reload:
            st.rerun()
        if do_start or do_stop:
            with st.spinner("Running script…"):
                try:
                    if do_start:
                        ok, output = control.start_llm(selected, stop_first=others)
                    else:
                        ok, output = control.stop_llm(selected)
                except control.ControlError as exc:
                    ok, output = False, str(exc)
            st.toast(f"{'start' if do_start else 'stop'} {selected.name}: {'ok' if ok else 'failed'}")
            if not ok and output:
                st.error(output[-2000:])
            time.sleep(1.5)
            st.rerun()
        st.markdown(f'<span style="font-size:11px;color:var(--faint)">{_html_escape(detail)}</span>',
                    unsafe_allow_html=True)

    decode = rates.decode_tps if rates and reachable else None
    prefill = rates.prefill_tps if rates and reachable else None
    history_points = list(rates.history) if rates and reachable else []
    with prefill_col:
        st.markdown(_throughput_col("Prefill", prefill, history_points, "prefill_tps"), unsafe_allow_html=True)
    with decode_col:
        st.markdown(_throughput_col("Decode", decode, history_points, "decode_tps"), unsafe_allow_html=True)
    with chart_col:
        st.markdown(
            '<div class="lr-legend">'
            '<span><span class="sw decode"></span><b>decode</b> tok/s</span>'
            '<span><span class="sw prefill"></span><b>prefill</b> tok/s</span>'
            '<span class="spacer"></span><span>last 5 min</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(_hero_chart(history_points), unsafe_allow_html=True)


def _selected_llm(names: list[str], status_map: dict[str, tuple[bool, str]]) -> config.LLMConfig:
    sel = st.session_state.get("model_sel")
    if sel not in names:
        running = [n for n in names if status_map[n][0]]
        sel = running[0] if running else names[0]
        st.session_state.model_sel = sel
    return next(llm for llm in llms if llm.name == sel)


def render_header() -> None:
    hc1, hc2 = st.columns([8, 1])
    with hc1:
        st.markdown(
            '<div class="lr-top">'
            '<span class="lr-brand"><span class="lr-logo">▦</span> DGX Spark <small>· LLM Runner</small></span>'
            '<span class="lr-live"><i></i> live · 2s</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with hc2:
        label = "☀" if st.session_state.theme == "dark" else "☾"
        if st.button(label, key="theme_toggle", help="Toggle light / dark", width="stretch"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()


def render_footer() -> None:
    fc1, fc2 = st.columns([8, 1])
    with fc1:
        st.markdown(
            '<div class="lr-foot"><span>DGX Spark · single-model mode · sampling every 2s</span></div>',
            unsafe_allow_html=True,
        )
    with fc2:
        st.page_link("pages/01_LLM_servers.py", label="LLM servers & logs ↗", width="content")


theme.render()
render_header()
render_hero()
render_hardware()
render_footer()
