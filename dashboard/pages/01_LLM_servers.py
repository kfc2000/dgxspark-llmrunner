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

LOG_TAIL_CHOICES = {
    "20": 20,
    "50": 50,
    "100": 100,
    "300": 300,
    "1000": 1000,
    "5000": 5000,
    "all (docker logs -f)": None,
}
LOG_VIEW_HEIGHT = 350


def view_logs(container: str, tail: int | None) -> None:
    try:
        logs = control.container_logs(container, tail)
    except control.ControlError as exc:
        st.error(str(exc))
        return
    st.caption(f"{len(logs.splitlines())} lines" + ("" if tail else " · full output, as `docker logs <name>`"))
    with st.container(height=LOG_VIEW_HEIGHT, autoscroll=True):
        st.code(logs or "(no output)", language="log")


def render_llm(llm: config.LLMConfig) -> None:
    with st.expander(f"{llm.name} — {llm.type}", expanded=True):
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
            do_start = b1.button("Start", key=f"start_{llm.name}", disabled=running, use_container_width=True)
            do_stop = b2.button("Stop", key=f"stop_{llm.name}", disabled=not running, use_container_width=True)
            do_refresh = b3.button("Refresh", key=f"refresh_{llm.name}", use_container_width=True)
            if do_refresh:
                st.rerun()
            if do_start or do_stop:
                action = control.start_llm if do_start else control.stop_llm
                with st.spinner("Running script…"):
                    try:
                        ok, output = action(llm)
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


for llm in llms:
    render_llm(llm)
