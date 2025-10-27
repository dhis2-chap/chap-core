# chap_core/assessment/metrics/vega_lite_dash.py

import altair as alt
import pandas as pd
from typing import Dict, Any, Optional
from chap_core.assessment.flat_representations import FlatObserved, FlatForecasts
import json
from chap_core.assessment.flat_representations import (
    convert_backtest_to_flat_forecasts,
    convert_backtest_observations_to_flat_observations,
    FlatMetric,
)
from chap_core.assessment.metrics import (
    DetailedRMSE,                   
    PeakValueDiffMetric,            
    PeakWeekLagMetric,              
    SamplesAboveTruthCountByTime,)  
from evaluation_plot import MetricByHorizonV2, MetricMapV2
from chap_core.database.tables import BackTest


def _compute_metric_df(metric_cls, flat_obs, flat_fc) -> pd.DataFrame:
    """Utility: run a MetricBase subclass and get its DataFrame result."""
    metric = metric_cls()
    return metric.compute(flat_obs, flat_fc)


def make_backtest_dashboard_from_backtest(
    flat_obs: FlatObserved,
    flat_fc: FlatForecasts,
    title: str = "Backtest dashboard",
    geojson: Optional[dict] = None,
) -> Dict[str, Any]:

    rmse_df = _compute_metric_df(DetailedRMSE, flat_obs, flat_fc)
    rmse_flat = FlatMetric(rmse_df)                                       

    peak_diff_df = _compute_metric_df(PeakValueDiffMetric, flat_obs, flat_fc)
    peak_diff_flat = FlatMetric(peak_diff_df)

    peak_lag_df = _compute_metric_df(PeakWeekLagMetric, flat_obs, flat_fc)

    above_cnt_time_df = _compute_metric_df(SamplesAboveTruthCountByTime, flat_obs, flat_fc)

    chart_rmse = MetricByHorizonV2(rmse_flat).plot()

    if geojson is not None and len(peak_diff_df) > 0:
        chart_second = MetricMapV2(peak_diff_flat, geojson=geojson).plot()
        chart_second = chart_second.properties(title="Peak value difference (truth - pred) per location")
    else:
        chart_second = MetricByHorizonV2(rmse_flat).plot().properties(
            title="(fallback) RMSE by horizon"
        )

    title_chart = alt.Chart().mark_text(align="left", fontSize=20, fontWeight="bold").encode(
        text=alt.value(title)
    ).properties(height=24)

    summary_df = (
        peak_diff_df.merge(peak_lag_df, on="location", how="outer", suffixes=("_diff", "_lag"))
        .fillna(0.0)
        .sort_values("location")
    )

    if len(above_cnt_time_df) > 0:
        loc_counts = (above_cnt_time_df.groupby("location")["metric"].sum().reset_index()
                      .rename(columns={"metric": "samples_above_cnt"}))
        summary_df = summary_df.merge(loc_counts, on="location", how="left").fillna(0.0)
    else:
        summary_df["samples_above_cnt"] = 0.0

    summary_df["summary"] = summary_df.apply(
        lambda r: (
            f"Lokasjon: {r['location']}"
            f"\nPeak diff: {r['metric_diff'] if 'metric_diff' in r else r.get('metric_x', 0):.2f}"
            f"\nPeak lag (uker): {r['metric_lag'] if 'metric_lag' in r else r.get('metric_y', 0):.0f}"
            f"\nSamples > truth (sum): {int(r['samples_above_cnt'])}"
        ),
        axis=1,
    )

    summary_chart = (
        alt.Chart(alt.Data(values=summary_df.to_dict(orient="records")))
        .mark_text(align="left", fontSize=13, lineBreak="\n")
        .encode(text="summary:N")
        .properties(height=min(200, 20 * max(1, len(summary_df))))
    )
    
    explainer_top = alt.Chart().mark_text(align="left", fontSize=12).encode(
        text=alt.value("Øverst: RMSE per horisont (lavere er bedre). Nederst: kart over peak-differanse per lokasjon.")
    ).properties(height=18)

    dashboard = alt.vconcat(
        title_chart,
        summary_chart,
        explainer_top,
        chart_rmse.properties(title="RMSE by horizon"),
        chart_second,
    ).configure(
        axis={"labelFontSize": 11, "titleFontSize": 12},
        legend={"labelFontSize": 11, "titleFontSize": 12},
        view={"stroke": None},
    )

    return dashboard

"""
if __name__ == "__main__":
    import json
    import pandas as pd

    obs_df = pd.read_csv("../../../../metrics/Assessment_example_chap_compatible/example_data/observations.csv")
    fc_df  = pd.read_csv("../../../../metrics/Assessment_example_chap_compatible/example_data/forecasts.csv")
    if "value" in obs_df.columns and "disease_cases" not in obs_df.columns:
        obs_df = obs_df.rename(columns={"value": "disease_cases"})
    if "time" in fc_df.columns and "time_period" not in fc_df.columns:
        fc_df = fc_df.rename(columns={"time": "time_period"})

    print("Obs columns:", list(obs_df.columns))
    print("Fc  columns:", list(fc_df.columns))

    flat_obs = FlatObserved(obs_df)
    flat_fc  = FlatForecasts(fc_df)

    # build chart/spec
    dashboard = make_backtest_dashboard_from_backtest(flat_obs, flat_fc)

    # if your function returns an Altair Chart, convert it to a dict first
    if hasattr(dashboard, "to_dict"):
        dashboard_spec = dashboard.to_dict(format="vega-lite")  # or "vega"
    else:
        dashboard_spec = dashboard  # already a dict

    # print full JSON
    print(json.dumps(dashboard_spec, indent=2, ensure_ascii=False))

    # save to file
    with open("backtest_dashboard_spec.json", "w", encoding="utf-8") as f:
        json.dump(dashboard_spec, f, indent=2, ensure_ascii=False)
    print("✅ wrote backtest_dashboard_spec.json")
"""


