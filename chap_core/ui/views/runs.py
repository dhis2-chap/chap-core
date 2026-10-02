"""History of every command run from the UI."""

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from chap_core.ui.services import (
    get_runs_dir,
    job_outputs,
    list_jobs,
    load_job,
    option_value,
    start_job,
    stop_job,
)
from chap_core.ui.widgets import STATUS, duration, job_title, open_in_results, page_header, short_name, show_outputs

STATUS_TEXT = {"running": "Running", "succeeded": "Succeeded", "failed": "Failed", "stopped": "Stopped"}
STATUS_STYLE = {"running": "color: #1F5FAD", "succeeded": "color: #1E7B45", "failed": "color: #B42318"}

page_header("Runs", "Everything started from the workbench. Runs keep going when you leave the page.")

jobs = list_jobs(get_runs_dir())
if not jobs:
    st.info("Nothing has been run yet.")
    st.stop()

left, right = st.columns([3, 2], gap="large")
with left:
    counts = {status: sum(job.status == status for job in jobs) for status in STATUS_TEXT}
    choices = ["All", *[s for s in STATUS_TEXT if counts[s]]]
    status_filter = st.segmented_control(
        "Show",
        choices,
        default="All",
        format_func=lambda s: f"All {len(jobs)}" if s == "All" else f"{STATUS_TEXT[s]} {counts[s]}",
        key="runs-filter",
    )
    shown = [job for job in jobs if status_filter in (None, "All") or job.status == status_filter]
    table = pd.DataFrame(
        {
            "Status": [STATUS_TEXT[job.status] for job in shown],
            "What": [job_title(job) for job in shown],
            "Dataset": [short_name(option_value(job.args, "--dataset-csv") or "") or "-" for job in shown],
            "Started": [job.started for job in shown],
            "Took": [duration(job) for job in shown],
        }
    )
    styled = table.style.apply(lambda col: [STATUS_STYLE.get(v.lower(), "") for v in col], subset=["Status"])
    selection = st.dataframe(
        styled,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="runs-table",
        height=min(36 * (len(shown) + 1) + 4, 640),
        column_config={"Started": st.column_config.DatetimeColumn(format="MMM D, HH:mm")},
    )
    rows = selection.selection.rows

if not shown:
    st.stop()
job = load_job(shown[rows[0] if rows else 0].run_dir)


def details(run_dir: Path, was_running: bool) -> None:
    job = load_job(run_dir)
    if was_running and job.status != "running":
        st.rerun()
    st.markdown(f"{STATUS[job.status]} :gray[in {duration(job)}]")
    st.subheader(job_title(job))
    dataset = option_value(job.args, "--dataset-csv")
    splits = option_value(job.args, "--backtest-params.n-splits")
    st.markdown(
        f":gray[{' · '.join(filter(None, [short_name(dataset) if dataset else None, f'{splits} splits' if splits else None, job.name]))}]"
    )
    buttons = st.container(horizontal=True)
    evaluations = [p for p in job_outputs(job) if p.suffix == ".nc"]
    if evaluations and buttons.button("Open in Results", type="primary", icon=":material/insights:"):
        open_in_results(evaluations[0])
    if job.status == "running":
        if buttons.button("Stop", icon=":material/stop:"):
            stop_job(job)
            st.rerun()
    else:
        if buttons.button("Run again", icon=":material/replay:"):
            start_job(get_runs_dir(), job.args, job.name.split("_", 1)[-1])
            st.rerun()
        if buttons.button("Delete", icon=":material/delete:"):
            shutil.rmtree(job.run_dir)
            st.rerun()
    log = job.log.read_text(errors="replace").splitlines() if job.log.exists() else []
    st.markdown("**Log, last lines**")
    st.code("\n".join(log[-5:]) or "(no output yet)", "log")
    with st.expander("Full log"):
        st.code("\n".join(log[-1000:]), "log", height=400)
    with st.expander("Show as CLI command"):
        st.code(job.command, "bash", wrap_lines=True)


with right, st.container(border=True, key="card-run-details"):
    if job.status == "running":
        st.fragment(run_every=2)(details)(job.run_dir, True)
    else:
        details(job.run_dir, False)

if job.status != "running":
    show_outputs(job)
