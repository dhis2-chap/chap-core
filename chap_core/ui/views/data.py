"""Pick, explore and validate a dataset (`chap validate`, `chap plot-dataset`)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from chap_core.plotting.dataset_plot import list_dataset_plots
from chap_core.ui.services import EXAMPLE_DATASETS, example_files, get_workdir, make_dataset_plot, save_upload
from chap_core.ui.widgets import settings

workdir = get_workdir()

st.title("Dataset")
st.caption(
    "Choose the CSV the model will be evaluated on: one row per location and time period, with "
    "`time_period`, `location`, `disease_cases` and covariates. A GeoJSON with the same name is picked up automatically."
)

examples = example_files(EXAMPLE_DATASETS)
uploads = sorted((workdir / "uploads").glob("*.csv"))
sources = ["Upload CSV", "Path or URL"]
if uploads:
    sources.insert(0, "Uploaded")
if examples:
    sources.insert(0, "Example dataset")
source = st.segmented_control("Source", sources, default=sources[0], key="data-source")

dataset_csv = None
if source == "Example dataset":
    dataset_csv = str(st.selectbox("Example", examples, format_func=lambda p: p.name))
elif source == "Uploaded":
    dataset_csv = str(st.selectbox("Uploaded file", uploads, format_func=lambda p: p.name))
elif source == "Upload CSV":
    csv_file = st.file_uploader("CSV file", type="csv")
    geojson_file = st.file_uploader("GeoJSON with region polygons (optional)", type=["geojson", "json"])
    if csv_file is not None:
        csv_path = save_upload(workdir, csv_file.name, csv_file.getvalue())
        if geojson_file is not None:
            save_upload(workdir, csv_path.with_suffix(".geojson").name, geojson_file.getvalue())
        dataset_csv = str(csv_path)
else:
    dataset_csv = st.text_input("Path or URL to a CSV file", value=st.session_state.get("dataset_csv", "")) or None

if dataset_csv is None:
    st.stop()

st.session_state["dataset_csv"] = dataset_csv
local = Path(dataset_csv).exists()

if local:
    df = pd.read_csv(dataset_csv)
    cols = st.columns(4)
    cols[0].metric("Rows", len(df))
    if "location" in df.columns:
        cols[1].metric("Locations", df["location"].nunique())
    if "time_period" in df.columns:
        cols[2].metric("From", str(df["time_period"].min()))
        cols[3].metric("To", str(df["time_period"].max()))
    geojson = Path(dataset_csv).with_suffix(".geojson")
    st.caption(f"Polygons: `{geojson.name}`" if geojson.exists() else "No GeoJSON next to this file.")

    overview_tab, plots_tab, table_tab = st.tabs(["Overview", "Plots", "Table"])
    with overview_tab:
        if {"time_period", "location", "disease_cases"} <= set(df.columns):
            st.line_chart(df.pivot_table(index="time_period", columns="location", values="disease_cases"))
        numeric = df.select_dtypes("number")
        st.dataframe(numeric.describe().T, width="stretch")
    with plots_tab:
        plots = {plot["id"]: plot for plot in list_dataset_plots()}
        plot_id = st.selectbox("Plot", list(plots), format_func=lambda i: plots[i]["name"])
        st.caption(plots[plot_id]["description"])
        if settings()["show_cli"]:
            st.code(f"chap plot-dataset {dataset_csv} --plot-name {plot_id}", "bash", wrap_lines=True)
        try:
            st.altair_chart(make_dataset_plot(Path(dataset_csv), plot_id), width="stretch")
        except Exception as e:
            st.warning(f"This plot is not available for this dataset: {e}")
    with table_tab:
        st.dataframe(df, height=400)

st.subheader("Validate")
model_name = st.text_input(
    "Also check against a model (optional)",
    value=st.session_state.get("model_name", ""),
    help="Local model directory or GitHub URL. Leave empty for a general check.",
)
if settings()["show_cli"]:
    st.code(
        f"chap validate --dataset-csv {dataset_csv}" + (f" --model-name {model_name}" if model_name else ""),
        "bash",
        wrap_lines=True,
    )
if st.button("Validate dataset", icon=":material/fact_check:"):
    from chap_core.cli_endpoints.validate import collect_validation_issues

    with st.spinner("Validating..."):
        try:
            issues = collect_validation_issues(dataset_csv, model_name or None)
        except Exception as e:
            st.exception(e)
            st.stop()
    if not issues:
        st.success("Validation passed: no issues found.")
    else:
        if any(issue.level == "error" for issue in issues):
            st.error("The dataset has errors that must be fixed before evaluation.")
        else:
            st.warning("The dataset has warnings.")
        st.dataframe(pd.DataFrame([issue.model_dump() for issue in issues]), hide_index=True)

if st.button("Next: evaluate a model", icon=":material/arrow_forward:"):
    st.switch_page("views/evaluate.py")
