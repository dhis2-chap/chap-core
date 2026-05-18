"""Side-by-side comparison plot for causal counterfactual analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots.evaluation_plot import EvaluationPlot

if TYPE_CHECKING:
    from chap_core.assessment.backtest_plots import ChartType


def _chart_for_location(evaluation, location: str, title: str) -> ChartType:
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
        )
        .properties(title=title)
    )


def plot_counterfactual(
    eval_original,
    eval_cf,
    counterfactual_columns: list[str] | None = None,
) -> ChartType:
    """Return a per-location vconcat of side-by-side Altair charts comparing original vs counterfactual."""
    locations = sorted(pd.DataFrame(eval_original.to_flat().observations)["location"].unique())

    rows = []
    for loc in locations:
        orig_chart = _chart_for_location(eval_original, loc, "Original")
        cf_chart = _chart_for_location(eval_cf, loc, "Counterfactual")
        rows.append(alt.hconcat(orig_chart, cf_chart).resolve_scale(y="shared"))

    subtitle = f" ({', '.join(counterfactual_columns)})" if counterfactual_columns else ""
    return alt.vconcat(*rows).properties(title=f"Causal Analysis: Original vs Counterfactual{subtitle}")
