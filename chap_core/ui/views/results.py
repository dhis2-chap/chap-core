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
from chap_core.ui.widgets import settings

workdir = get_workdir()

st.title("Results")
st.caption("Plot and compare evaluation files written by **Evaluate** and other commands.")

uploaded = st.file_uploader("Add evaluation files (.nc)", type="nc", accept_multiple_files=True)
uploads = [save_upload(workdir, f.name, f.getvalue()) for f in uploaded or []]
uploads += sorted((workdir / "uploads").glob("*.nc"))

options = list(
    dict.fromkeys(str(p) for p in [*uploads, *list_evaluations(workdir), *example_files(EXAMPLE_EVALUATIONS)])
)
if not options:
    st.info("No evaluations yet. Run one on the Evaluate page or upload a .nc file.")
    st.stop()


def label(path: str) -> str:
    p = Path(path)
    if p.is_relative_to(workdir / "runs"):
        run = p.relative_to(workdir / "runs").parts[0]
        return run if p.name == "evaluation.nc" else f"{run}/{p.name}"
    return p.name


default = [p for p in st.session_state.get("selected_evals", []) if p in options] or options[:1]
selected = st.multiselect("Evaluations", options, default=default, format_func=label)
if not selected:
    st.stop()
st.session_state["selected_evals"] = selected
st.session_state["evaluation"] = selected[0]


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
        if settings()["show_cli"]:
            st.code(f"chap plot-backtest {shlex.quote(path)} plot.html --plot-type {plot_id}", "bash", wrap_lines=True)
        try:
            st.altair_chart(cached_plot(path, Path(path).stat().st_mtime, plot_id), width="stretch")
        except Exception as e:
            st.exception(e)

with metrics_tab:
    if settings()["show_cli"]:
        st.code(
            shlex.join(["chap", "export-metrics", "--input-files", *selected, "--output-file", "metrics.csv"]),
            "bash",
            wrap_lines=True,
        )
    try:
        metrics = cached_metrics(tuple(selected), tuple(Path(p).stat().st_mtime for p in selected))
    except Exception as e:
        st.warning(f"Metrics could not be computed for this selection: {e}")
        st.stop()
    metrics = metrics.assign(filename=[label(p) for p in selected])
    st.dataframe(metrics, hide_index=True)
    st.download_button("Download CSV", metrics.to_csv(index=False), "metrics.csv", "text/csv")
