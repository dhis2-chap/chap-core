"""Side-by-side comparison plot for causal counterfactual analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots.evaluation_plot import EvaluationPlot

if TYPE_CHECKING:
    from chap_core.assessment.backtest_plots import ChartType


def plot_counterfactual(
    eval_original,
    eval_cf,
    counterfactual_columns: list[str] | None = None,
) -> ChartType:
    """Return a side-by-side Altair chart comparing original vs counterfactual predictions."""

    def _chart(evaluation, title: str):
        flat = evaluation.to_flat()
        historical = pd.DataFrame(flat.historical_observations) if flat.historical_observations is not None else None
        return (
            EvaluationPlot()
            .plot(pd.DataFrame(flat.observations), pd.DataFrame(flat.forecasts), historical)
            .properties(title=title)
        )

    subtitle = f" ({', '.join(counterfactual_columns)})" if counterfactual_columns else ""
    return alt.hconcat(
        _chart(eval_original, "Original"),
        _chart(eval_cf, "Counterfactual"),
    ).properties(title=f"Causal Analysis: Original vs Counterfactual{subtitle}")
