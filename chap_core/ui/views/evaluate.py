"""Configure and run a backtest evaluation (`chap eval`)."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from chap_core.ui.commands import command_fields
from chap_core.ui.models import model_kind, model_label
from chap_core.ui.services import EXAMPLE_MODEL, backtest_windows
from chap_core.ui.widgets import (
    GLOBAL_RUN_OPTIONS,
    advanced_sections,
    field_rows,
    field_widget,
    job_details,
    page_header,
    recent_runs,
    run_panel,
    settings,
    short_name,
)

FORM = "eval"
BACKTEST = ["backtest_params.n_periods", "backtest_params.n_splits", "backtest_params.stride"]


@st.cache_data(show_spinner=False)
def dataset_overview(path: str, mtime: float) -> dict:
    from chap_core.cli_endpoints.validate import collect_validation_issues

    df = pd.read_csv(path)
    try:
        issues = collect_validation_issues(path)
    except Exception as e:
        issues = [type("Issue", (), {"level": "error", "message": str(e)})()]
    return {
        "periods": sorted(df["time_period"].astype(str).unique()) if "time_period" in df else [],
        "locations": df["location"].nunique() if "location" in df else 0,
        "errors": sum(issue.level == "error" for issue in issues),
        "warnings": sum(issue.level == "warning" for issue in issues),
    }


def card_header(title: str, popover_label: str):
    cols = st.columns([3, 1], vertical_alignment="center")
    cols[0].subheader(title)
    return cols[1].popover(popover_label, width="stretch")


page_header(
    "Evaluate a model",
    "Train on expanding windows of the data and forecast the periods that follow, "
    "so every forecast can be checked against what actually happened.",
)

fields = {f.key: f for f in command_fields(FORM)}
prefill = (
    {"model_name": str(EXAMPLE_MODEL)} if EXAMPLE_MODEL.exists() and not st.session_state.get("model_name") else {}
)
values = {key: settings()[key] for key in GLOBAL_RUN_OPTIONS if key in fields}

left, right = st.columns([2, 1], gap="large")
with left:
    with st.container(border=True, key="card-eval-model"):
        with card_header("Model", "Change model"):
            values["model_name"] = field_widget(fields["model_name"], FORM, prefill)
        model = values["model_name"]
        if model:
            st.markdown(f"**{model_label(model)}** :gray[· {model_kind(model)}]")
            st.caption(f"`{model}`")
        else:
            st.markdown(":gray[No model chosen yet.]")
        values["model_configuration_yaml"] = field_widget(fields["model_configuration_yaml"], FORM, prefill)

    with st.container(border=True, key="card-eval-dataset"):
        with card_header("Dataset", "Change dataset"):
            values["dataset_csv"] = field_widget(fields["dataset_csv"], FORM, prefill)
        dataset = values["dataset_csv"]
        overview = (
            dataset_overview(dataset, Path(dataset).stat().st_mtime) if dataset and Path(dataset).exists() else None
        )
        if overview:
            problems = overview["errors"] or overview["warnings"]
            facts = {
                "File": short_name(dataset),
                "Locations": overview["locations"],
                "Periods": f"{overview['periods'][0]} to {overview['periods'][-1]}" if overview["periods"] else "-",
                "Validation": ":green[No issues]"
                if not problems
                else f":red[{overview['errors']} errors, {overview['warnings']} warnings]",
            }
            for col, (name, value) in zip(st.columns(4), facts.items(), strict=True):
                col.markdown(f":gray[{name}]  \n**{value}**")
        elif dataset:
            st.caption(f"`{dataset}`")
        else:
            st.markdown(":gray[No dataset chosen yet. Pick one on the Data page or with Change dataset.]")

    with st.container(border=True, key="card-eval-backtest"):
        st.subheader("Backtest")
        values |= field_rows([fields[key] for key in BACKTEST], FORM, prefill)
        n_periods, n_splits, stride = (values[key] or fields[key].cli_default for key in BACKTEST)
        if overview and overview["periods"]:
            periods = overview["periods"]
            windows = backtest_windows(periods, n_periods, n_splits, stride)
            if not windows:
                st.warning("The dataset is too short for this many splits and periods.")
            else:
                st.caption(
                    f"{n_splits} forecasts of {n_periods} periods for {short_name(dataset)}, "
                    f"the first starting {windows[0]['forecast_start']}."
                )
                index = {period: i for i, period in enumerate(periods)}
                bars = pd.DataFrame(
                    [
                        row
                        for w in windows
                        for row in (
                            {
                                "Split": f"Split {w['split']}",
                                "start": 0,
                                "end": index[w["train_end"]] + 1,
                                "Window": "Training data",
                                "From": periods[0],
                                "To": w["train_end"],
                            },
                            {
                                "Split": f"Split {w['split']}",
                                "start": index[w["forecast_start"]],
                                "end": index[w["forecast_end"]] + 1,
                                "Window": "Forecast window",
                                "From": w["forecast_start"],
                                "To": w["forecast_end"],
                            },
                        )
                    ]
                )
                labels = "[" + ",".join(f"'{p}'" for p in periods) + "]"
                chart = (
                    alt.Chart(bars)
                    .mark_bar(height=10, cornerRadius=2)
                    .encode(
                        x=alt.X(
                            "start:Q",
                            title=None,
                            scale=alt.Scale(domain=[0, len(periods)]),
                            axis=alt.Axis(labelExpr=f"{labels}[datum.value] || ''", tickCount=6),
                        ),
                        x2="end:Q",
                        y=alt.Y("Split:N", title=None, sort=None, axis=alt.Axis(labelOverlap=False)),
                        color=alt.Color(
                            "Window:N",
                            scale=alt.Scale(domain=["Training data", "Forecast window"], range=["#C9D6E6", "#1F5FAD"]),
                            legend=alt.Legend(orient="bottom", title=None),
                        ),
                        tooltip=["Split", "Window", "From", "To"],
                    )
                    .properties(height=alt.Step(26))
                )
                st.altair_chart(chart, width="stretch")

    rest = [
        f
        for key, f in fields.items()
        if key not in values and key not in GLOBAL_RUN_OPTIONS and key not in {"model_name", "dataset_csv"}
    ]
    values |= advanced_sections(rest, FORM, prefill)

with right:
    model = values.get("model_name")
    summary = {
        "Model": model_label(model) if model else "-",
        "Dataset": short_name(values["dataset_csv"]) if values.get("dataset_csv") else "-",
        "Forecasts": f"{n_splits} splits x {n_periods} periods",
        "Output": f"`{values.get('output_file') or 'evaluation.nc'}`",
    }
    job = run_panel(
        FORM,
        list(fields.values()),
        values,
        label=f"eval {model_label(model) if model else 'model'}",
        summary=summary,
        run_label="Run evaluation",
    )
    recent_runs()

if job is not None:
    job_details(job)
