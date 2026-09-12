from __future__ import annotations

import streamlit as st

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0b0e14",
        "panel": "#12161f",
        "panel-2": "#161b26",
        "menu-bg": "#141926",
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
        "scheme": "dark",
    },
    "light": {
        "bg": "#f3f5f9",
        "panel": "#ffffff",
        "panel-2": "#ffffff",
        "menu-bg": "#ffffff",
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
        "scheme": "light",
    },
}

_CSS = """
<style>
html { __VARS__ }
html { color-scheme: __SCHEME__; }
html, body, .stApp, [data-testid="stAppContainer"], [data-testid="stAppView"],
[data-testid="stAppContainer"] > div, [data-testid="stBottomBlockContainer"] {
  background-color: var(--bg) !important;
}
body {
  background-image:
    radial-gradient(1200px 500px at 80% -10%, var(--glow-1), transparent 60%),
    radial-gradient(900px 420px at -10% 110%, var(--glow-2), transparent 60%);
  color: var(--text) !important;
  font-variant-numeric: tabular-nums;
  transition: background-color .25s, color .25s;
}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background: var(--panel) !important; border-right: 1px solid var(--border) !important; color: var(--text);
}
[data-testid="stSidebar"] [data-testid="stNavItem"], [data-testid="stSidebar"] a { color: var(--muted) !important; }
[data-testid="stSidebar"] [data-testid="stNavItem"]:hover { background: var(--hover) !important; color: var(--text) !important; }
[data-testid="stSidebar"] [data-testid="stNavItem"][aria-current="true"],
[data-testid="stSidebar"] li[data-active="true"] > a { color: var(--accent) !important; }
[data-baseweb="menu"] { background: var(--menu-bg) !important; color: var(--text) !important; }
header[data-testid="stHeader"] { display: none; }
footer { display: none; }
.block-container { max-width: 1500px; margin: 0 auto; padding-top: 14px; padding-bottom: 18px; }

[data-testid="stButton"] button {
  background: var(--panel-2); color: var(--text); border: 1px solid var(--border-hi);
  border-radius: 10px; font-weight: 600;
  height: 32px; min-height: 32px !important; max-height: 32px; padding: 4px 14px;
}
[data-testid="stButton"] button:hover { border-color: var(--hover-border); background: var(--hover); color: var(--text); }
[data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent), #3a6fe0);
  border-color: transparent; color: #fff; box-shadow: 0 4px 16px rgba(79,140,255,.3);
}
[data-testid="stSelectbox"] > div,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
  background: var(--panel) !important; color: var(--text) !important;
  border: 1px solid var(--border-hi) !important; border-radius: 10px !important;
}
[data-testid="stSelectbox"] select {
  background: transparent !important; color: var(--text) !important;
  font-weight: 650; font-size: 15.5px;
}
[data-testid="stSelectbox"] option { background: var(--panel) !important; color: var(--text) !important; }
[data-testid="stTextInput"] input, [data-testid="stTextInput"] div {
  background: var(--panel) !important; color: var(--text) !important;
}
[data-testid="stCaption"] { color: var(--faint) !important; }
[data-testid="stAlert"] { background: var(--panel); border-color: var(--border-hi); }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stWidgetLabel"] { color: var(--muted) !important; }
code, pre, [data-testid="stCode"], [data-testid="stCodeBlockContainer"] {
  background: var(--bg) !important; color: var(--text) !important;
}
[data-testid="stCodeBlockContainer"] { border: 1px solid var(--border) !important; border-radius: 10px; }
[data-testid="stCodeBlockCopyButton"] { color: var(--faint) !important; }
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div > div[data-testid="stBottomBlockContainer"] {
  background: var(--bg) !important;
}
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
  height: 22px; box-sizing: content-box; margin-top: 3px;
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


def render() -> None:
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    palette = PALETTES[st.session_state.theme]
    var_decl = " ".join(f"--{k}:{v};" for k, v in palette.items() if k != "scheme")
    css = _CSS.replace("__VARS__", var_decl).replace("__SCHEME__", palette["scheme"])
    st.markdown(css, unsafe_allow_html=True)
