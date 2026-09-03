"""Side-by-side and overlaid comparison plots for causal counterfactual analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd

from chap_core.assessment.backtest_plots.evaluation_plot import EvaluationPlot
from chap_core.plotting.backtest_plot import clean_time

if TYPE_CHECKING:
    from chap_core.assessment.backtest_plots import ChartType

_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
_ORIGINAL_LABEL = "Original"
_COUNTERFACTUAL_LABEL = "Counterfactual"
_OBSERVED_LABEL = "Observed (pre-split)"
_DEFAULT_X_LABEL = "Time period"
_DEFAULT_Y_LABEL = "Disease cases"


def _mark_type(layer) -> str | None:
    mark = getattr(layer, "mark", None)
    return getattr(mark, "type", mark)


def _set_axis_titles(chart: ChartType, x_label: str, y_label: str) -> None:
    """Set axis titles on every layer of a layered chart.

    Vega-Lite ignores `title` in axis config, so `configure_axisX` / `configure_axisY`
    cannot be used to label the axes. Every layer needs the same explicit title, or
    Vega-Lite merges the per-layer field names into titles like "q_10, q_25".
    """
    for layer in chart.layer:
        encoding = getattr(layer, "encoding", None)
        for channel, label in (("x", x_label), ("y", y_label)):
            channel_def = getattr(encoding, channel, None)
            if channel_def is not None and channel_def is not alt.Undefined:
                channel_def.title = label


def _chart_for_location(
    flat_evaluation,
    location: str,
    title: str,
    *,
    x_label: str,
    y_label: str,
    show_confidence_intervals: bool = True,
) -> ChartType:
    obs = pd.DataFrame(flat_evaluation.observations)
    forecasts = pd.DataFrame(flat_evaluation.forecasts)
    historical = None
    if flat_evaluation.historical_observations is not None:
        hist = pd.DataFrame(flat_evaluation.historical_observations)
        loc_hist = hist[hist["location"] == location]
        historical = loc_hist if not loc_hist.empty else None
    chart = (
        EvaluationPlot()
        .plot(
            obs[obs["location"] == location],
            forecasts[forecasts["location"] == location],
            historical,
        )
        .properties(title=title)
    )
    if not show_confidence_intervals:
        chart.layer = [layer for layer in chart.layer if _mark_type(layer) != "errorband"]
    _set_axis_titles(chart, x_label, y_label)
    return chart


def plot_counterfactual(
    eval_original,
    eval_cf,
    counterfactual_columns: list[str] | None = None,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    original_label: str = _ORIGINAL_LABEL,
    counterfactual_label: str = _COUNTERFACTUAL_LABEL,
    show_confidence_intervals: bool = True,
) -> ChartType:
    """Return a per-location vconcat of side-by-side Altair charts comparing original vs counterfactual."""
    x_label = x_label if x_label is not None else _DEFAULT_X_LABEL
    y_label = y_label if y_label is not None else _DEFAULT_Y_LABEL
    locations = sorted(pd.DataFrame(eval_original.to_flat().observations)["location"].unique())
    flat_evaluation = eval_original.to_flat()
    flat_evaluation_cf = eval_cf.to_flat()

    rows = []
    for loc in locations:
        orig_chart = _chart_for_location(
            flat_evaluation,
            loc,
            original_label,
            x_label=x_label,
            y_label=y_label,
            show_confidence_intervals=show_confidence_intervals,
        )
        cf_chart = _chart_for_location(
            flat_evaluation_cf,
            loc,
            counterfactual_label,
            x_label=x_label,
            y_label=y_label,
            show_confidence_intervals=show_confidence_intervals,
        )
        rows.append(alt.hconcat(orig_chart, cf_chart).resolve_scale(y="shared"))

    return alt.vconcat(*rows).properties(
        title=_compose_title(title, counterfactual_columns, original_label, counterfactual_label)
    )


def _compose_title(
    title: str | None,
    counterfactual_columns: list[str] | None,
    original_label: str,
    counterfactual_label: str,
) -> str:
    if title is not None:
        return title
    subtitle = f" ({', '.join(counterfactual_columns)})" if counterfactual_columns else ""
    return f"Causal Analysis: {original_label} vs {counterfactual_label}{subtitle}"


def _location_quantiles(forecasts: pd.DataFrame, location: str, series: str) -> pd.DataFrame:
    """Per-time-period forecast quantiles for one location, tagged with a series label."""
    subset = forecasts[forecasts["location"] == location]
    grouped = subset.groupby("time_period")["forecast"].quantile(np.array(_QUANTILES)).unstack()
    grouped.columns = [f"q_{int(q * 100)}" for q in _QUANTILES]
    grouped = grouped.reset_index()
    grouped["time_period"] = grouped["time_period"].astype(str).apply(clean_time)
    grouped["series"] = series
    return grouped


def _location_observed(flat_evaluation, location: str) -> pd.DataFrame | None:
    """Observed disease_cases for one location, restricted to the pre-split historical window."""
    if flat_evaluation.historical_observations is None:
        return None
    df = pd.DataFrame(flat_evaluation.historical_observations)
    df = df[df["location"] == location]
    if df.empty:
        return None
    df = df.copy()
    df["time_period"] = df["time_period"].astype(str).apply(clean_time)
    df["series"] = _OBSERVED_LABEL
    return df


def _x(x_label: str) -> alt.X:
    return alt.X("time_period:T", title=x_label)


def _y(field: str, y_label: str) -> alt.Y:
    return alt.Y(f"{field}:Q", scale=alt.Scale(zero=False), title=y_label)


def _color(color_scale: alt.Scale) -> alt.Color:
    return alt.Color("series:N", scale=color_scale, legend=alt.Legend(title=None))


def _overlay_for_location(
    quantiles: pd.DataFrame,
    observed: pd.DataFrame | None,
    location: str,
    x_label: str,
    y_label: str,
    color_scale: alt.Scale,
    show_confidence_intervals: bool = True,
) -> ChartType:
    median_line = (
        alt.Chart(quantiles).mark_line().encode(x=_x(x_label), y=_y("q_50", y_label), color=_color(color_scale))
    )
    layers = []
    if show_confidence_intervals:
        layers.append(
            alt.Chart(quantiles)
            .mark_errorband(opacity=0.15)
            .encode(x=_x(x_label), y=_y("q_10", y_label), y2="q_90:Q", color=_color(color_scale))
        )
        layers.append(
            alt.Chart(quantiles)
            .mark_errorband(opacity=0.3)
            .encode(x=_x(x_label), y=_y("q_25", y_label), y2="q_75:Q", color=_color(color_scale))
        )
    layers.append(median_line)
    if observed is not None and not observed.empty:
        layers.append(
            alt.Chart(observed)
            .mark_line(strokeDash=[4, 2])
            .encode(x=_x(x_label), y=_y("disease_cases", y_label), color=_color(color_scale))
        )
    return alt.layer(*layers).properties(title=location)


def plot_counterfactual_overlayed(
    eval_original,
    eval_cf,
    counterfactual_columns: list[str] | None = None,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    original_label: str = _ORIGINAL_LABEL,
    counterfactual_label: str = _COUNTERFACTUAL_LABEL,
    show_confidence_intervals: bool = True,
) -> ChartType:
    """Return a per-location vconcat of single overlaid charts: original vs counterfactual forecasts
    on shared axes, with the observed line drawn only up to the split point."""
    x_label = x_label if x_label is not None else _DEFAULT_X_LABEL
    y_label = y_label if y_label is not None else _DEFAULT_Y_LABEL
    flat_original = eval_original.to_flat()
    flat_cf = eval_cf.to_flat()
    original_forecasts = pd.DataFrame(flat_original.forecasts)
    cf_forecasts = pd.DataFrame(flat_cf.forecasts)
    locations = sorted(original_forecasts["location"].unique())

    color_scale = alt.Scale(
        domain=[original_label, counterfactual_label, _OBSERVED_LABEL],
        range=["#1f77b4", "#d62728", "#333333"],
    )

    rows = []
    for loc in locations:
        quantiles = pd.concat(
            [
                _location_quantiles(original_forecasts, loc, original_label),
                _location_quantiles(cf_forecasts, loc, counterfactual_label),
            ],
            ignore_index=True,
        )
        observed = _location_observed(flat_original, loc)
        rows.append(
            _overlay_for_location(quantiles, observed, loc, x_label, y_label, color_scale, show_confidence_intervals)
        )

    return alt.vconcat(*rows).properties(
        title=_compose_title(title, counterfactual_columns, original_label, counterfactual_label)
    )
