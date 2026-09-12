from __future__ import annotations

import math
import time

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import streamlit as st

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

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0b0e14",
        "panel": "#12161f",
        "panel-2": "#161b26",
        "border": "#1f2635",
        "border-hi": "#2c3547",
        "text": "#e6ebf3",
        "muted": "#8a93a6",
        "faint": "#5b6373",
        "accent": "#4f8cff",
        "accent-2": "#8b5cf6",
        "ok": "#2dd4a7",
        "warn": "#ffb454",
        "err": "#ff5d73",
        "hover": "#1b2231",
        "hover-border": "#3d4c6b",
        "glow-1": "rgba(79,140,255,.09)",
        "glow-2": "rgba(139,92,246,.07)",
    },
    "light": {
        "bg": "#f3f5f9",
        "panel": "#ffffff",
        "panel-2": "#ffffff",
        "border": "#e2e7ef",
        "border-hi": "#ccd5e2",
        "text": "#17202f",
        "muted": "#5a6474",
        "faint": "#8a93a3",
        "accent": "#2f6fed",
        "accent-2": "#7a45e0",
        "ok": "#0da27d",
        "warn": "#c97a10",
        "err": "#d63a58",
        "hover": "#eef1f7",
        "hover-border": "#b7c4d8",
        "glow-1": "rgba(47,111,237,.07)",
        "glow-2": "rgba(122,69,224,.05)",
    },
}

_CSS = """
<style>
html, body, [data-testid="stAppContainer"] { __VARS__ }
body {
  background-color: var(--bg) !important;
  background-image:
    radial-gradient(1200px 500px at 80% -10%, var(--glow-1), transparent 60%),
    radial-gradient(900px 420px at -10% 110%, var(--glow-2), transparent 60%) !important;
  color: var(--text) !important;
  font-variant-numeric: tabular-nums;
  transition: background-color .25s, color .25s;
}
header[data-testid="stHeader"] { display: none; }
footer { display: none; }
.block-container { max-width: 1500px; margin: 0 auto; padding-top: 14px; padding-bottom: 18px; }

[data-testid="stButton"] button {
  background: var(--panel-2); color: var(--text); border: 1px solid var(--border-hi);
  border-radius: 10px; font-weight: 600;
}
[data-testid="stButton"] button:hover { border-color: var(--hover-border); background: var(--hover); color: var(--text); }
[data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent), #3a6fe0);
  border-color: transparent; color: #fff; box-shadow: 0 4px 16px rgba(79,140,255,.3);
}
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stSelectbox"] select {
  background: var(--panel) !important; color: var(--text) !important;
  border: 1px solid var(--border-hi) !important; border-radius: 10px !important;
}
[data-testid="stSelectbox"] select { font-weight: 650; font-size: 15.5px; }
[data-testid="stCaption"] { color: var(--faint) !important; }
[data-testid="stAlert"] { background: var(--panel); border-color: var(--border-hi); }
a { color: var(--muted); }
a:hover { color: var(--accent); }

.lr-top { display: flex; align-items: center; gap: 14px; }
.lr-brand { display: flex; align-items: center; gap: 10px; font-weight: 650; font-size: 17px; letter-spacing: .2px; }
.lr-brand small { color: var(--muted); font-weight: 450; }
.lr-logo {
  width: 30px; height: 30px; border-radius: 9px; display: inline-grid; place-items: center;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: #fff; font-size: 15px; box-shadow: 0 4px 18px rgba(79,140,255,.35);
}
.lr-live {
  display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted);
  border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px; background: var(--panel);
}
.lr-live i {
  width: 7px; height: 7px; border-radius: 50%; background: var(--ok);
  box-shadow: 0 0 8px var(--ok); animation: lrpulse 2s infinite;
}
@keyframes lrpulse { 50% { opacity: .35; } }

.lr-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 10px; }
.lr-row3 { grid-template-columns: repeat(3, 1fr); }
.lr-card {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  border: 1px solid var(--border); border-radius: 14px;
  padding: 10px 12px 6px; transition: border-color .2s;
}
.lr-card:hover { border-color: var(--border-hi); }
.lr-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 2px; }
.lr-label {
  font-size: 11.5px; font-weight: 600; letter-spacing: .8px;
  text-transform: uppercase; color: var(--muted);
}
.lr-sub { font-size: 11.5px; color: var(--faint); }
.lr-value { font-size: 21px; font-weight: 680; letter-spacing: -.5px; line-height: 1.15; }
.lr-value small { font-size: 12.5px; font-weight: 500; color: var(--muted); margin-left: 2px; }
.lr-spark { display: block; width: 100%; height: 38px; margin-top: 4px; }

[data-testid="stHorizontalContainer"] {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  border: 1px solid var(--border); border-radius: 14px;
}
[data-testid="stHorizontalContainer"] > [data-testid="stColumn"]:nth-child(-n+3) {
  border-right: 1px solid var(--border);
}
.lr-t { display: flex; flex-direction: column; justify-content: center; height: 100%; padding: 6px 14px; }
.lr-t .lr-value { font-size: 24px; line-height: 1.05; }
.lr-delta { font-size: 11.5px; font-weight: 600; min-height: 17px; }
.lr-delta.up { color: var(--ok); }
.lr-delta.down { color: var(--err); }

.lr-status {
  display: inline-flex; align-items: center; gap: 8px; align-self: flex-start;
  font-size: 12px; font-weight: 650; letter-spacing: .6px; padding: 4px 11px; border-radius: 999px;
}
.lr-status i { width: 8px; height: 8px; border-radius: 50%; }
.lr-status.running { color: var(--ok); background: color-mix(in srgb, var(--ok) 10%, transparent); border: 1px solid color-mix(in srgb, var(--ok) 40%, transparent); }
.lr-status.running i { background: var(--ok); box-shadow: 0 0 10px var(--ok); }
.lr-status.starting { color: var(--warn); background: color-mix(in srgb, var(--warn) 10%, transparent); border: 1px solid color-mix(in srgb, var(--warn) 40%, transparent); }
.lr-status.starting i { background: var(--warn); box-shadow: 0 0 10px var(--warn); }
.lr-status.stopped { color: var(--faint); background: var(--bg); border: 1px solid var(--border); }
.lr-status.stopped i { background: var(--faint); }

.lr-legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
.lr-legend b { font-weight: 600; color: var(--text); }
.lr-legend .spacer { flex: 1; }
.lr-legend .sw { display: inline-block; width: 18px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
.sw.decode { background: var(--accent); }
.sw.prefill { background: var(--warn); }
.lr-chart svg { width: 100%; height: 74px; display: block; }

.lr-foot { display: flex; justify-content: space-between; align-items: center; color: var(--faint); font-size: 12px; }

@media (max-width: 1100px) {
  .lr-row, .lr-row3 { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""


def _render_css() -> None:
    palette = PALETTES[st.session_state.theme]
    var_decl = " ".join(f"--{k}:{v};" for k, v in palette.items())
    st.markdown(_CSS.replace("__VARS__", var_decl), unsafe_allow_html=True)


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
        f"{'M' if i == 0 else 'L'}{i * step:.1f},{SPARK_H - 2 - (min(max(v, lo), hi) - lo) / (hi - lo) * (SPARK_H - 4):.1f}"
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
        parts.append(f'<path fill="url(#gDec)" d="{dec} L700,150 L{max((dec_x0 - (now - WINDOW_S)) / WINDOW_S * 700, 0):.0f},150 Z"/>')
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
        pill = ("running", "RUNNING") if (running and reachable) else (("starting", "STARTING") if running else ("stopped", "STOPPED"))
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


_render_css()
render_header()
render_hero()
render_hardware()
render_footer()
