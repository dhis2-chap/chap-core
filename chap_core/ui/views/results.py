"""Plot and compare evaluation results (`chap plot-backtest`, `chap export-metrics`)."""

import shlex
from pathlib import Path

import streamlit as st

from chap_core.assessment.backtest_plots import list_backtest_plots
from chap_core.cli_endpoints.utils import compute_metrics_table
from chap_core.ui.services import (
    EXAMPLE_EVALUATIONS,
    example_files,
    get_workdir,
    list_evaluations,
    make_plot,
    save_upload,
)

workdir = get_workdir()

st.title("Results")

uploaded = st.file_uploader("Add evaluation files (.nc)", type="nc", accept_multiple_files=True)
uploads = [save_upload(workdir, f.name, f.getvalue()) for f in uploaded or []]

options = [str(p) for p in [*uploads, *list_evaluations(workdir), *example_files(EXAMPLE_EVALUATIONS)]]
if not options:
    st.info("No evaluations yet. Run one on the Evaluate page or upload a .nc file.")
    st.stop()


def label(path: str) -> str:
    p = Path(path)
    return p.parent.name if p.parent.parent.name == "runs" else p.name


default = [p for p in st.session_state.get("selected_evals", []) if p in options] or options[:1]
selected = st.multiselect("Evaluations", options, default=default, format_func=label)
if not selected:
    st.stop()


@st.cache_resource(show_spinner="Building plot...")
def cached_plot(path: str, mtime: float, plot_id: str):
    return make_plot(Path(path), plot_id)


@st.cache_data(show_spinner="Computing metrics...")
def cached_metrics(paths: tuple[str, ...], mtimes: tuple[float, ...]):
    return compute_metrics_table([Path(p) for p in paths])


plots_tab, metrics_tab = st.tabs(["Plots", "Metrics"])

with plots_tab:
    plots = {plot["id"]: plot for plot in list_backtest_plots()}
    plot_id = st.selectbox("Plot type", list(plots), format_func=lambda i: plots[i]["name"])
    st.caption(plots[plot_id]["description"])
    for path in selected:
        st.subheader(label(path))
        st.code(f"chap plot-backtest {shlex.quote(path)} plot.html --plot-type {plot_id}", "bash")
        try:
            st.altair_chart(cached_plot(path, Path(path).stat().st_mtime, plot_id), width="stretch")
        except Exception as e:
            st.exception(e)

with metrics_tab:
    st.code(
        shlex.join(["chap", "export-metrics", "--input-files", *selected, "--output-file", "metrics.csv"]),
        "bash",
    )
    metrics = cached_metrics(tuple(selected), tuple(Path(p).stat().st_mtime for p in selected)).assign(
        filename=[label(p) for p in selected]
    )
    st.dataframe(metrics, hide_index=True)
    st.download_button("Download CSV", metrics.to_csv(index=False), "metrics.csv", "text/csv")
