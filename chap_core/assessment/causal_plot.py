"""Side-by-side comparison plot for causal counterfactual analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots.evaluation_plot import EvaluationPlot

if TYPE_CHECKING:
    from chap_core.assessment.backtest_plots import ChartType


def _location_y_domain(eval_original, eval_cf, location: str) -> list[float]:
    """Compute shared y-domain for one location across both evaluations.

    The lower bound is the minimum of all observations (including historical).
    The upper bound is the maximum of all observations and q_90 of forecast samples,
    so that outlier raw samples do not inflate the axis.
    """
    obs_values: list[pd.Series] = []
    upper_values: list[pd.Series] = []

    for ev in (eval_original, eval_cf):
        flat = ev.to_flat()

        forecasts = pd.DataFrame(flat.forecasts)
        loc_fc = forecasts[forecasts["location"] == location].dropna(subset=["forecast"])
        if not loc_fc.empty:
            per_period_q90 = loc_fc.groupby("time_period")["forecast"].quantile(0.9)
            upper_values.append(pd.Series([float(per_period_q90.max())]))

        for df_raw in (flat.observations, flat.historical_observations):
            if df_raw is None:
                continue
            df = pd.DataFrame(df_raw)
            if "location" in df.columns and "disease_cases" in df.columns:
                vals = df.loc[df["location"] == location, "disease_cases"].dropna()
                if not vals.empty:
                    obs_values.append(vals)
                    upper_values.append(vals)

    if not obs_values or not upper_values:
        return [0.0, 1.0]
    all_lower = pd.concat(obs_values)
    all_upper = pd.concat(upper_values)
    return [float(all_lower.min()), float(all_upper.max())]


def _chart_for_location(evaluation, location: str, y_domain: list[float], title: str) -> ChartType:
    flat = evaluation.to_flat()
    obs = pd.DataFrame(flat.observations)
    forecasts = pd.DataFrame(flat.forecasts)
    historical = None
    if flat.historical_observations is not None:
        hist = pd.DataFrame(flat.historical_observations)
        loc_hist = hist[hist["location"] == location]
        historical = loc_hist if not loc_hist.empty else None
    return (
        EvaluationPlot()
        .plot(
            obs[obs["location"] == location],
            forecasts[forecasts["location"] == location],
            historical,
            y_domain=y_domain,
        )
        .properties(title=title)
    )


def plot_counterfactual(
    eval_original,
    eval_cf,
    counterfactual_columns: list[str] | None = None,
) -> ChartType:
    """Return a per-location vconcat of side-by-side Altair charts comparing original vs counterfactual.

    For each location the two panels share the same y-axis domain so predictions
    and observations are directly comparable.
    """
    locations = sorted(pd.DataFrame(eval_original.to_flat().observations)["location"].unique())

    rows = []
    for loc in locations:
        y_domain = _location_y_domain(eval_original, eval_cf, loc)
        orig_chart = _chart_for_location(eval_original, loc, y_domain, "Original")
        cf_chart = _chart_for_location(eval_cf, loc, y_domain, "Counterfactual")
        rows.append(alt.hconcat(orig_chart, cf_chart))

    subtitle = f" ({', '.join(counterfactual_columns)})" if counterfactual_columns else ""
    return alt.vconcat(*rows).properties(title=f"Causal Analysis: Original vs Counterfactual{subtitle}")
