"""Pick, explore and validate a dataset (`chap validate`, `chap plot-dataset`)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from chap_core.plotting.dataset_plot import list_dataset_plots
from chap_core.ui.maps import choropleth_url
from chap_core.ui.services import (
    EXAMPLE_DATASETS,
    dataset_geojson,
    dataset_incidence,
    example_files,
    get_workdir,
    make_dataset_plot,
    save_upload,
)
from chap_core.ui.widgets import page_header

workdir = get_workdir()
page_header(
    "Dataset",
    "One row per location and time period, with time_period, location, disease_cases and covariates. "
    "A GeoJSON with the same name is picked up automatically.",
)

with st.container(border=True, key="card-data-source"):
    st.subheader("Choose a dataset")
    examples = example_files(EXAMPLE_DATASETS)
    uploads = sorted((workdir / "uploads").glob("*.csv"))
    sources = [s for s, ok in (("Example", examples), ("Uploaded", uploads)) if ok] + ["Upload", "Path or URL"]
    source = st.pills("Source", sources, default=sources[0], key="data-source", label_visibility="collapsed")
    dataset_csv = None
    if source == "Example":
        dataset_csv = str(st.selectbox("Example dataset", examples, format_func=lambda p: p.name))
    elif source == "Uploaded":
        dataset_csv = str(st.selectbox("Uploaded file", uploads, format_func=lambda p: p.name))
    elif source == "Upload":
        cols = st.columns(2)
        csv_file = cols[0].file_uploader("CSV file", type="csv")
        geojson_file = cols[1].file_uploader("Region polygons (optional)", type=["geojson", "json"])
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

if Path(dataset_csv).exists():
    df = pd.read_csv(dataset_csv)
    with st.container(border=True, key="card-data-overview"):
        cols = st.columns(4)
        cols[0].metric("Rows", len(df))
        if "location" in df.columns:
            cols[1].metric("Locations", df["location"].nunique())
        if "time_period" in df.columns:
            cols[2].metric("From", str(df["time_period"].min()))
            cols[3].metric("To", str(df["time_period"].max()))
        polygons = Path(dataset_csv).with_suffix(".geojson")
        st.caption(f"Polygons from `{polygons.name}`" if polygons.exists() else "No GeoJSON next to this file.")
        geojson = dataset_geojson(Path(dataset_csv))
        overview_tab, map_tab, plots_tab, table_tab = st.tabs(["Cases", "Map", "Plots", "Table"])
        with map_tab:
            if geojson is None:
                st.info("Add a GeoJSON with the same name as the CSV to see the regions on a map.")
            else:
                try:
                    values, legend = dataset_incidence(Path(dataset_csv))
                    st.iframe(choropleth_url(geojson, values, legend), height=480)
                except Exception as e:
                    st.warning(f"The map is not available for this dataset: {e}")
        with overview_tab:
            if {"time_period", "location", "disease_cases"} <= set(df.columns):
                st.line_chart(df.pivot_table(index="time_period", columns="location", values="disease_cases"))
            st.dataframe(df.select_dtypes("number").describe().T, width="stretch")
        with plots_tab:
            plots = {plot["id"]: plot for plot in list_dataset_plots()}
            plot_id = st.selectbox("Plot", list(plots), format_func=lambda i: plots[i]["name"])
            st.caption(plots[plot_id]["description"])
            try:
                st.altair_chart(make_dataset_plot(Path(dataset_csv), plot_id), width="stretch")
            except Exception as e:
                st.warning(f"This plot is not available for this dataset: {e}")
            with st.expander("Show as CLI command"):
                st.code(f"chap plot-dataset {dataset_csv} --plot-name {plot_id}", "bash", wrap_lines=True)
        with table_tab:
            st.dataframe(df, height=400)

with st.container(border=True, key="card-data-validate"):
    st.subheader("Validate")
    model_name = st.text_input(
        "Also check against a model (optional)",
        value=st.session_state.get("model_name", ""),
        help="Local model directory or GitHub URL. Leave empty for a general check.",
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
    with st.expander("Show as CLI command"):
        command = f"chap validate --dataset-csv {dataset_csv}" + (f" --model-name {model_name}" if model_name else "")
        st.code(command, "bash", wrap_lines=True)

if st.button("Next: evaluate a model", type="primary", icon=":material/arrow_forward:"):
    st.switch_page("views/evaluate.py")
