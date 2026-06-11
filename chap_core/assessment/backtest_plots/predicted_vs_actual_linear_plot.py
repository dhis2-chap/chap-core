"""
Predicted vs actual scatter plot in linear space.

Same layout as the log1p plot — horizon-faceted scatter colored by location
with a dashed identity reference line — but on raw linear axes, which makes
over- and under-prediction at high case counts immediately visible.
"""

import pandas as pd

from chap_core.assessment.backtest_plots import ChartType, backtest_plot
from chap_core.assessment.backtest_plots.predicted_vs_actual_plot import (
    PredictedVsActualPlot,
    build_predicted_vs_actual_chart,
)


@backtest_plot(
    plot_id="predicted_vs_actual_linear",
    name="Predicted vs Actual (linear)",
    description="Scatter plots of predicted (median) vs actual values on linear axes, faceted by horizon and colored by location.",
)
class PredictedVsActualLinearPlot(PredictedVsActualPlot):
    """Linear-axes variant of :class:`PredictedVsActualPlot`.

    Inherits the horizon faceting and the median-vs-observed preprocessing; only
    the per-facet chart differs (raw values instead of log1p-transformed axes).
    """

    def _plot(self, data: pd.DataFrame) -> ChartType:
        return build_predicted_vs_actual_chart(
            data,
            x_field="median_forecast",
            y_field="disease_cases",
            title="Predicted vs Actual (linear scale)",
        )
