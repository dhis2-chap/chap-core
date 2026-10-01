"""Compare evaluation files (`chap plot-backtest`, `chap export-metrics`)."""

import shlex
from pathlib import Path

import pandas as pd
import streamlit as st

from chap_core.assessment.backtest_plots import list_backtest_plots
from chap_core.cli_endpoints.utils import compute_metrics_table
from chap_core.ui.maps import choropleth_url
from chap_core.ui.services import (
    EXAMPLE_EVALUATIONS,
    dataset_geojson,
    evaluation_dataset,
    example_files,
    get_workdir,
    list_evaluations,
    list_jobs,
    make_plot,
    metric_by_location,
    plot_facets,
    save_upload,
)
from chap_core.ui.widgets import job_title, page_header

# Headline metrics: error metrics where lower is better, and the share of observations inside the 50% interval.
HEADLINE = {"crps": "CRPS", "mae": "MAE", "rmse": "RMSE", "coverage_25_75": "Within 50% interval"}

workdir = get_workdir()
page_header("Results", "Compare evaluations side by side. Lower is better for every error metric.")

titles = {
    str(p): job_title(job) for job in list_jobs(workdir) if job.status == "succeeded" for p in job.run_dir.glob("*.nc")
}
uploads = sorted((workdir / "uploads").glob("*.nc"))
options = list(
    dict.fromkeys(str(p) for p in [*list_evaluations(workdir), *uploads, *example_files(EXAMPLE_EVALUATIONS)])
)


def label(path: str) -> str:
    p = Path(path)
    if path in titles:
        run = p.parent.name.split("_")[0]
        return f"{titles[path]} · {run}" if p.name == "evaluation.nc" else f"{titles[path]} · {p.name}"
    return p.name


with st.container(horizontal=True, vertical_alignment="bottom"):
    default = [p for p in st.session_state.get("selected_evals", []) if p in options] or options[:1]
    selected = st.multiselect("Comparing", options, default=default, format_func=label, key="results-selected")
    with st.popover("Add evaluation file", icon=":material/upload:"):
        for upload in st.file_uploader("Evaluation files (.nc)", type="nc", accept_multiple_files=True) or []:
            path = str(save_upload(workdir, upload.name, upload.getvalue()))
            st.session_state["selected_evals"] = [*selected, path]
            st.rerun()

if not selected:
    st.info("Pick evaluations to compare, run one on the Evaluate page, or add a file.")
    st.stop()
st.session_state["selected_evals"] = selected
st.session_state["evaluation"] = selected[0]


@st.cache_data(show_spinner="Computing metrics...")
def metrics_for(paths: tuple[str, ...], mtimes: tuple[float, ...]) -> pd.DataFrame:
    return compute_metrics_table([Path(p) for p in paths])


@st.cache_resource(show_spinner="Building plot...")
def plot_for(path: str, mtime: float, plot_id: str, coords: tuple):
    return make_plot(Path(path), plot_id, dict(coords))


@st.cache_data(show_spinner=False)
def facets_for(path: str, mtime: float, plot_id: str):
    return plot_facets(Path(path), plot_id)


def mtime(path: str) -> float:
    return Path(path).stat().st_mtime


with st.container(border=True, key="card-metrics"):
    st.subheader("Metrics")
    metrics: pd.DataFrame | None
    try:
        metrics = metrics_for(tuple(selected), tuple(mtime(p) for p in selected))
    except Exception as e:
        st.warning(f"Metrics could not be computed for this selection: {e}")
        metrics = None
    if metrics is not None:
        metrics.insert(0, "Evaluation", [label(p) for p in selected])
        headline = metrics[["Evaluation", *[c for c in HEADLINE if c in metrics]]].rename(columns=HEADLINE)
        best = {name: headline[name].min() for name in headline.columns[1:] if name != HEADLINE["coverage_25_75"]}
        if HEADLINE["coverage_25_75"] in headline:
            coverage = headline[HEADLINE["coverage_25_75"]]
            best[HEADLINE["coverage_25_75"]] = coverage.loc[(coverage - 0.5).abs().idxmin()]
        styled = headline.style.format(precision=3).apply(
            lambda col: ["font-weight: 700" if col.name in best and v == best[col.name] else "" for v in col]
        )
        st.dataframe(styled, hide_index=True, width="stretch")
        cols = st.columns([3, 1], vertical_alignment="center")
        cols[0].caption("Bold marks the best value in each column; for the 50% interval, the one closest to 50%.")
        cols[1].download_button(
            "Download CSV", metrics.to_csv(index=False), "metrics.csv", "text/csv", icon=":material/download:"
        )
        with st.expander("All metrics"):
            st.dataframe(metrics, hide_index=True)
        with st.expander("Show as CLI command"):
            st.code(
                shlex.join(["chap", "export-metrics", "--input-files", *selected, "--output-file", "metrics.csv"]),
                "bash",
                wrap_lines=True,
            )


@st.cache_data(show_spinner="Computing metrics per location...")
def location_metric(path: str, mtime: float, metric_id: str) -> dict[str, float]:
    return metric_by_location(Path(path), metric_id)


with st.container(border=True, key="card-map"):
    st.subheader("Map")
    controls = st.container(horizontal=True, vertical_alignment="bottom", gap="medium")
    map_eval = controls.selectbox("Evaluation", selected, format_func=label, key="results-map-eval")
    metric_id = controls.selectbox(
        "Metric", list(HEADLINE), format_func=lambda m: HEADLINE[m], key="results-map-metric"
    )
    dataset = evaluation_dataset(Path(map_eval))
    geojson = dataset_geojson(Path(dataset)) if dataset and Path(dataset).exists() else None
    if geojson is None:
        st.caption(
            "No region polygons for this evaluation. Maps are shown for evaluations run here "
            "on a dataset with a GeoJSON next to it."
        )
    else:
        try:
            values = location_metric(map_eval, mtime(map_eval), metric_id)
            hint = "share inside the 50% interval" if metric_id == "coverage_25_75" else "lower is better"
            st.iframe(choropleth_url(geojson, values, f"{HEADLINE[metric_id]} ({hint})"), height=480)
        except Exception as e:
            st.warning(f"The map is not available for this evaluation: {e}")

with st.container(border=True, key="card-plot"):
    plots = {plot["id"]: plot for plot in list_backtest_plots()}
    controls = st.container(horizontal=True, vertical_alignment="bottom", gap="medium")
    plot_id = controls.selectbox("Plot", list(plots), format_func=lambda i: plots[i]["name"], key="results-plot")
    st.caption(plots[plot_id]["description"])
    facets = facets_for(selected[0], mtime(selected[0]), plot_id)
    coords = {}
    for column, display, values in facets:
        choice = controls.selectbox(display, values, key=f"results-facet-{column}") if len(values) > 1 else values[0]
        coords[column] = choice
    for path in selected:
        if len(selected) > 1:
            st.markdown(f"**{label(path)}**")
        try:
            chart = plot_for(path, mtime(path), plot_id, tuple(coords.items()))
            st.altair_chart(chart, width="stretch" if coords else "content")
        except Exception as e:
            st.warning(f"This plot is not available for {label(path)}: {e}")
    with st.expander("Show as CLI command"):
        st.code(
            f"chap plot-backtest {shlex.quote(selected[0])} plot.html --plot-type {plot_id}", "bash", wrap_lines=True
        )
