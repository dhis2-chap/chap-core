# chap_core/assessment/metrics/vega_lite_dash.py

import altair as alt
import pandas as pd
from typing import Optional
import json

from chap_core import get_temp_dir
from chap_core.assessment.flat_representations import FlatObserved, FlatForecasts
from chap_core.assessment.flat_representations import (
    FlatMetric,
)
from chap_core.assessment.metrics.above_truth import RatioOfSamplesAboveTruth
from chap_core.plotting.evaluation_plot import MetricByHorizonV2Mean, MetricByTimePeriodAndLocationV2Mean


def _compute_metric_df(metric_cls, flat_obs, flat_fc) -> pd.DataFrame:
    """Utility: run a MetricBase subclass and get its DataFrame result."""
    metric = metric_cls()
    return metric.compute(flat_obs, flat_fc)


def _horizon_section(flat_obs: FlatObserved, flat_fc: FlatForecasts, geojson: Optional[dict]):
    # rmse_df = _compute_metric_df(DetailedRMSE, flat_obs, flat_fc)
    # peak_diff_df = _compute_metric_df(PeakValueDiffMetric, flat_obs, flat_fc)
    # peak_lag_df = _compute_metric_df(PeakPeriodLagMetric, flat_obs, flat_fc)
    above_truth_df = _compute_metric_df(RatioOfSamplesAboveTruth, flat_obs, flat_fc)
    print("Above truth df:", above_truth_df)

    # rmse_flat = FlatMetric(rmse_df)
    # peak_diff_flat = FlatMetric(peak_diff_df)
    # peak_lag_flat = FlatMetric(peak_lag_df)
    above_truth_flat = FlatMetric(above_truth_df)

    charts = []
    # charts.append(MetricByHorizonV2Mean(rmse_flat).plot().properties(
    #         title={
    #             "text": "RMSE by horizon distance",
    #             "subtitle": "Lower is better; measures average forecast error magnitude across samples. x-axis: forecast horizon distance",
    #             "anchor": "start",
    #             "fontSize": 16,
    #             "subtitleFontSize": 12
    #         }))
    # charts.append(MetricByHorizonV2Mean(peak_diff_flat).plot().properties(
    #     title={
    #         "text": "Peak value difference (truth - pred) by horizon",
    #         "subtitle": "Negative = earlier peak (model before truth); Positive = later peak (model after truth). x-axis: forecast horizon distance",
    #         "anchor": "start",
    #         "fontSize": 16,
    #         "subtitleFontSize": 12
    #     }))
    # charts.append(MetricByHorizonV2Mean(peak_lag_flat).plot().properties(
    #     title={
    #         "text": "Peak week lag (weeks) by horizon",
    #         "subtitle": "Negative = overprediction (model peak > truth); Positive = underprediction (model peak < truth). x-axis: forecast horizon distance",
    #         "anchor": "start",
    #         "fontSize": 16,
    #         "subtitleFontSize": 12
    #     }))
    charts.append(
        MetricByHorizonV2Mean(above_truth_flat)
        .plot()
        .properties(
            title={
                "text": "Samples above truth by horizon",
                "subtitle": "Count of samples where forecast > truth. x-axis: forecast horizon distance",
                "anchor": "start",
                "fontSize": 16,
                "subtitleFontSize": 12,
            }
        )
    )

    # if geojson is not None and len(peak_diff_df) > 0:
    #     charts.append(MetricMapV2(peak_diff_flat, geojson=geojson).plot().properties(
    #         title="Peak value difference map"))

    # section_title = (
    #     alt.Chart().mark_text(align="left", fontSize=16, fontWeight="bold")
    #     .encode(text=alt.value("By horizon")).properties(height=20)
    # )
    # return [section_title] + charts
    return charts


def _time_section(flat_obs: FlatObserved, flat_fc: FlatForecasts):
    # rmse_df = _compute_metric_df(DetailedRMSE, flat_obs, flat_fc)
    # peak_diff_df = _compute_metric_df(PeakValueDiffMetric, flat_obs, flat_fc)
    # peak_lag_df = _compute_metric_df(PeakPeriodLagMetric, flat_obs, flat_fc)
    above_truth_df = _compute_metric_df(RatioOfSamplesAboveTruth, flat_obs, flat_fc)

    # Ensure columns needed for grouping exist
    # rmse_flat = FlatMetric(rmse_df)
    # peak_diff_flat = FlatMetric(peak_diff_df)
    # peak_lag_flat = FlatMetric(peak_lag_df)
    above_truth_flat = FlatMetric(above_truth_df)

    charts = []
    # charts.append(MetricByTimePeriodV2Mean(rmse_flat).plot().properties(
    #         title={
    #             "text": "RMSE by time period",
    #             "subtitle": "Lower is better; measures average forecast error magnitude across samples. x-axis: time period of observation",
    #             "anchor": "start",
    #             "fontSize": 16,
    #             "subtitleFontSize": 12
    #         }))
    # charts.append(MetricByTimePeriodV2Mean(peak_diff_flat).plot().properties(
    #         title={
    #             "text": "Peak value difference (truth - pred) by time period",
    #             "subtitle": "Negative = overprediction (model peak > truth); Positive = underprediction (model peak < truth). x-axis: true peak week",
    #             "anchor": "start",
    #             "fontSize": 16,
    #             "subtitleFontSize": 12
    #         }))
    # charts.append(MetricByTimePeriodV2Mean(peak_lag_flat).plot().properties(
    #         title={
    #             "text": "Peak week lag (weeks) by time period",
    #             "subtitle": "Negative = earlier peak (model before truth); Positive = later peak (model after truth). x-axis: peak observation week",
    #             "anchor": "start",
    #             "fontSize": 16,
    #             "subtitleFontSize": 12
    #         }))
    charts.append(
        MetricByTimePeriodAndLocationV2Mean(above_truth_flat)
        .plot()
        .properties(
            title={
                "text": "Samples above truth by time period",
                "subtitle": "Count of samples where forecast > truth. x-axis: time period of observation",
                "anchor": "start",
                "fontSize": 16,
                "subtitleFontSize": 12,
            }
        )
    )

    # section_title = (
    #     alt.Chart().mark_text(align="left", fontSize=16, fontWeight="bold")
    #     .encode(text=alt.value("By time period")).properties(height=20)
    # )
    # return [section_title] + charts
    return charts


# --- NEW: combined dashboard --------------------------------------------
def combined_dashboard_from_backtest(
    flat_obs: FlatObserved,
    flat_fc: FlatForecasts,
    title: str = "Backtest dashboard",
    geojson: Optional[dict] = None,
) -> alt.Chart:
    charts = []
    # title
    charts.append(
        alt.Chart()
        .mark_text(align="left", fontSize=20, fontWeight="bold")
        .encode(text=alt.value(title))
        .properties(height=24)
    )

    # Add some text
    charts.append(
        alt.Chart()
        .mark_text(align="left", fontSize=12)
        .encode(
            text=alt.value(
                "These plots show biases in the samples returned by the model, "
                "\nspecifically whether the samples generally are over or under the true observation.\n"
                "This can be used to assess model calibration and tendency to over- or under-predict."
            )
        )
        .properties(height=18)
    )

    # sections
    charts += _horizon_section(flat_obs, flat_fc, geojson)
    charts += _time_section(flat_obs, flat_fc)

    dashboard = alt.vconcat(*charts).configure(
        axis={"labelFontSize": 11, "titleFontSize": 12},
        legend={"labelFontSize": 11, "titleFontSize": 12},
        view={"stroke": None},
    )
    return dashboard


if __name__ == "__main__":
    obs_df = pd.read_csv("../../../../metrics/Assessment_example_chap_compatible/example_data/observations.csv")
    fc_df = pd.read_csv("../../../../metrics/Assessment_example_chap_compatible/example_data/forecasts.csv")
    if "value" in obs_df.columns and "disease_cases" not in obs_df.columns:
        obs_df = obs_df.rename(columns={"value": "disease_cases"})
    if "time" in fc_df.columns and "time_period" not in fc_df.columns:
        fc_df = fc_df.rename(columns={"time": "time_period"})

    print("Obs columns:", list(obs_df.columns))
    print("Fc  columns:", list(fc_df.columns))

    flat_obs = FlatObserved(obs_df)
    flat_fc = FlatForecasts(fc_df)

    # build chart/spec
    dashboard = combined_dashboard_from_backtest(flat_obs, flat_fc, title="Backtest dashboard")

    dashboard_spec = dashboard.to_dict(format="vega-lite")

    print(json.dumps(dashboard_spec, indent=2, ensure_ascii=False))
    output_path = get_temp_dir() / "backtest_dashboard_spec.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_spec, f, indent=2, ensure_ascii=False)
