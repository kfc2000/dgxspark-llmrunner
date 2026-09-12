from __future__ import annotations

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import streamlit as st

from llmrunner import config, control, llm_metrics
from llmrunner.config import ConfigError

st.set_page_config(page_title="LLM servers", page_icon="⚙", layout="wide")

tracker = llm_metrics.get_tracker()

try:
    llms = config.load_config()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

st.title("LLM servers")
st.caption(f"Config: `{config.DEFAULT_CONFIG_PATH}`")

if not llms:
    st.info("No LLMs configured.")
    st.stop()

if "llm_action_result" not in st.session_state:
    st.session_state.llm_action_result = {}

running = control.running_llms(llms)
if running:
    names = ", ".join(llm.name for llm in running)
    st.warning(f"**Single-model mode:** `{names}` {'is' if len(running) == 1 else 'are'} loaded. "
               "Starting another model will stop it first.")
else:
    st.info("**Single-model mode:** no model loaded. The DGX Spark hosts one LLM at a time.")

active = running[0].name if running else llms[0].name
choice = st.selectbox("Model", [llm.name for llm in llms], index=[llm.name for llm in llms].index(active))
selected = next(llm for llm in llms if llm.name == choice)

LOG_TAIL_CHOICES = {
    "20": 20,
    "50": 50,
    "100": 100,
    "300": 300,
}
LOG_VIEW_HEIGHT = 350


def view_logs(container: str, tail: int) -> None:
    try:
        logs = control.container_logs(container, tail)
    except control.ControlError as exc:
        st.error(str(exc))
        return
    st.caption(f"{len(logs.splitlines())} lines")
    with st.container(height=LOG_VIEW_HEIGHT, autoscroll=True):
        st.code(logs or "(no output)", language="log")


def render_llm(llm: config.LLMConfig, others_running: list[config.LLMConfig]) -> None:
    st.subheader(f"{llm.name} — {llm.type}")
    status_col, action_col = st.columns([2, 3])
    with status_col:
        running, detail = control.is_llm_running(llm)
        (st.success if running else st.error)(f"{'RUNNING' if running else 'STOPPED'} — {detail}")
        st.caption(f"workdir: `{llm.workdir}`")
        st.caption(f"endpoint: `{llm.endpoint}`")
        if llm.container:
            st.caption(f"container: `{llm.container}`")

    with action_col:
        b1, b2, b3 = st.columns(3)
        swap = bool(others_running)
        label = "Stop others & Start" if swap else "Start"
        do_start = b1.button(
            label,
            key=f"start_{llm.name}",
            disabled=running,
            type="primary" if swap else "secondary",
            width="stretch",
        )
        do_stop = b2.button("Stop", key=f"stop_{llm.name}", disabled=not running, width="stretch")
        do_refresh = b3.button("Refresh", key=f"refresh_{llm.name}", width="stretch")
        if swap:
            st.caption(f"Will stop {', '.join(f'`{o.name}`' for o in others_running)} before starting.")
        if do_refresh:
            st.rerun()
        if do_start or do_stop:
            with st.spinner("Running script…"):
                try:
                    if do_start:
                        ok, output = control.start_llm(llm, stop_first=others_running)
                    else:
                        ok, output = control.stop_llm(llm)
                except control.ControlError as exc:
                    ok, output = False, str(exc)
            st.session_state.llm_action_result[llm.name] = (ok, output, "start" if do_start else "stop")
            st.rerun()

    result = st.session_state.llm_action_result.get(llm.name)
    if result:
        ok, output, action = result
        (st.toast if ok else st.error)(f"{action} {llm.name}: {'ok' if ok else 'failed'}")
        if output:
            (st.code if ok else st.error)(output[-4000:])

    rates = tracker.scrape(llm.name, llm.endpoint)
    if rates.reachable:
        m1, m2, m3 = st.columns(3)
        m1.metric("Decode speed", f"{rates.decode_tps:.1f} tok/s" if rates.decode_tps is not None else "n/a")
        m2.metric("Prefill speed", f"{rates.prefill_tps:.1f} tok/s" if rates.prefill_tps is not None else "n/a")
        m3.metric(
            "Running requests", f"{rates.running_requests:.0f}" if rates.running_requests is not None else "n/a"
        )

    if llm.container:
        tail = st.selectbox("Log lines", list(LOG_TAIL_CHOICES), index=0, key=f"tail_{llm.name}")
        auto = st.toggle("Auto refresh (5s)", value=True, key=f"autologs_{llm.name}")

        def log_view(tail_label: str = tail, container: str = llm.container) -> None:
            view_logs(container, LOG_TAIL_CHOICES[tail_label])

        if auto:
            st.fragment(log_view, run_every="5s")()
        else:
            log_view()
    else:
        st.caption("Set `container` in llms.json to view docker logs.")


render_llm(selected, [o for o in running if o.name != selected.name])
