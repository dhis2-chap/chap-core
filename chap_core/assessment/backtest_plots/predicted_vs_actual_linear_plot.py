"""
Predicted vs actual scatter plot in linear space.

Same layout as the log1p plot — horizon-faceted scatter colored by location
with a dashed identity reference line — but on raw linear axes, which makes
over- and under-prediction at high case counts immediately visible.
"""

import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot
from chap_core.assessment.backtest_plots.predicted_vs_actual_plot import (
    build_predicted_vs_actual_chart,
    median_forecasts_joined_with_observations,
)


@backtest_plot(
    plot_id="predicted_vs_actual_linear",
    name="Predicted vs Actual (linear)",
    description="Scatter plots of predicted (median) vs actual values on linear axes, faceted by horizon and colored by location.",
)
class PredictedVsActualLinearPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        merged = median_forecasts_joined_with_observations(forecasts, observations)
        return build_predicted_vs_actual_chart(
            merged,
            x_field="median_forecast",
            y_field="disease_cases",
            title="Predicted vs Actual (linear scale)",
        )
