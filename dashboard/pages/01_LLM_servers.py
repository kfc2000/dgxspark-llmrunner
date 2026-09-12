from __future__ import annotations

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import streamlit as st
import theme

from llmrunner import config, control
from llmrunner.config import ConfigError

st.set_page_config(page_title="LLM servers", page_icon="⚙", layout="wide")
theme.render()

try:
    llms = config.load_config()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

st.title("LLM servers")

if not llms:
    st.info("No LLMs configured.")
    st.stop()

names = [llm.name for llm in llms]
choice = st.selectbox("Model", names, key="srv_model")
selected = next(llm for llm in llms if llm.name == choice)

st.subheader(selected.name)

LOG_TAIL_CHOICES = {"20": 20, "50": 50, "100": 100, "300": 300}
LOG_VIEW_HEIGHT = 350

if not selected.container:
    st.caption("Set `container` in llms.json to view docker logs.")
    st.stop()


tail = st.selectbox("Log lines", list(LOG_TAIL_CHOICES), index=2, key="srv_tail")
auto = st.toggle("Auto refresh (5s)", value=True, key="srv_auto")


def log_view(tail_label: str = tail, container: str = str(selected.container)) -> None:
    try:
        logs = control.container_logs(container, LOG_TAIL_CHOICES[tail_label])
    except control.ControlError as exc:
        st.error(str(exc))
        return
    st.caption(f"{len(logs.splitlines())} lines")
    with st.container(height=LOG_VIEW_HEIGHT, autoscroll=True):
        st.code(logs or "(no output)", language="log")


if auto:
    st.fragment(log_view, run_every="5s")()
else:
    log_view()
