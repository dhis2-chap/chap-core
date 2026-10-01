"""History of every command run from the UI."""

import itertools
import shutil

import pandas as pd
import streamlit as st

from chap_core.ui.services import get_workdir, list_jobs, load_job
from chap_core.ui.widgets import STATUS_ICONS, job_panel

st.title("Runs")
st.caption("Every command started from the UI, with its log and outputs. Runs keep going if you leave the page.")

jobs = list_jobs(get_workdir())
if not jobs:
    st.info("Nothing has been run yet.")
    st.stop()

status_filter = st.segmented_control(
    "Status", ["running", "succeeded", "failed", "stopped"], selection_mode="multi", default=None
)
shown = [job for job in jobs if not status_filter or job.status in status_filter]

table = pd.DataFrame(
    {
        "Status": [job.status for job in shown],
        "Run": [job.name for job in shown],
        "Command": [" ".join(itertools.takewhile(lambda a: not a.startswith("-"), job.args)) for job in shown],
        "Started": [job.started for job in shown],
        "Seconds": [((job.finished or pd.Timestamp.now().to_pydatetime()) - job.started).seconds for job in shown],
    }
)
selection = st.dataframe(
    table,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="runs-table",
    column_config={"Started": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")},
)
rows = selection.selection.rows
if not rows:
    st.caption("Select a run to see its log and outputs.")
    st.stop()

job = load_job(shown[rows[0]].run_dir)
st.subheader(f"{STATUS_ICONS[job.status]} {job.name}")
st.code(job.command, "bash", wrap_lines=True)
job_panel(job)
if job.status != "running" and st.button("Delete run", icon=":material/delete:"):
    shutil.rmtree(job.run_dir)
    st.rerun()
