"""Pick, preview and validate a dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st

from chap_core.ui.services import EXAMPLE_DATASETS, example_files, get_workdir, save_upload

workdir = get_workdir()

st.title("Dataset")
st.caption("Choose the CSV the model will be evaluated on. A GeoJSON with the same name is picked up automatically.")

examples = example_files(EXAMPLE_DATASETS)
sources = ["Upload CSV", "Path or URL"]
if examples:
    sources.insert(0, "Example dataset")
source = st.radio("Source", sources, horizontal=True)

dataset_csv = None
if source == "Example dataset":
    dataset_csv = str(st.selectbox("Example", examples, format_func=lambda p: p.name))
elif source == "Upload CSV":
    csv_file = st.file_uploader("CSV file", type="csv")
    geojson_file = st.file_uploader("GeoJSON with region polygons (optional)", type=["geojson", "json"])
    if csv_file is not None:
        csv_path = save_upload(workdir, csv_file.name, csv_file.getvalue())
        if geojson_file is not None:
            save_upload(workdir, csv_path.with_suffix(".geojson").name, geojson_file.getvalue())
        dataset_csv = str(csv_path)
else:
    dataset_csv = st.text_input("Path or URL to a CSV file") or None

if dataset_csv is None:
    st.stop()

st.session_state["dataset_csv"] = dataset_csv
st.success(f"Selected dataset: `{dataset_csv}`")

if Path(dataset_csv).exists():
    df = pd.read_csv(dataset_csv)
    cols = st.columns(3)
    cols[0].metric("Rows", len(df))
    if "location" in df.columns:
        cols[1].metric("Locations", df["location"].nunique())
    if "time_period" in df.columns:
        cols[2].metric("Periods", f"{df['time_period'].min()} to {df['time_period'].max()}")
    if {"time_period", "location", "disease_cases"} <= set(df.columns):
        st.line_chart(df.pivot_table(index="time_period", columns="location", values="disease_cases"))
    st.dataframe(df, height=250)

st.subheader("Validate")
model_name = st.text_input(
    "Also check against a model (optional)",
    help="Local model directory or GitHub URL. Leave empty for a general check.",
)
st.code(f"chap validate --dataset-csv {dataset_csv}" + (f" --model-name {model_name}" if model_name else ""), "bash")
if st.button("Validate dataset"):
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
        st.dataframe(pd.DataFrame([issue.model_dump() for issue in issues]))

st.page_link("views/evaluate.py", label="Next: evaluate a model", icon=":material/arrow_forward:")
