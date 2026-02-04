"""
Metrics dashboard plot for backtests.

This module provides a backtest plot that shows various metrics by forecast
horizon and time period.
"""

from typing import Optional

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.assessment.metrics import (
    Coverage25_75Metric,
    CRPSNormMetric,
    RMSEMetric,
)
from chap_core.plotting.backtest_plot import text_chart, title_chart
from chap_core.plotting.evaluation_plot import MetricByHorizonV2Mean, MetricByTimePeriodV2Mean


@backtest_plot(
    id="metrics_dashboard",
    name="Overview of various metrics by horizon/time",
    description="A dashboard showing various metrics by forecast horizon and time period.",
)
class MetricsDashboard(BacktestPlotBase):
    """
    Backtest plot (as prototype/example) with various metrics.
    """

    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: Optional[pd.DataFrame] = None,
    ) -> ChartType:
        """
        Generate and return the dashboard visualization.

        Parameters
        ----------
        observations : pd.DataFrame
            Observed values with columns: location, time_period, disease_cases
        forecasts : pd.DataFrame
            Forecast samples with columns: location, time_period, horizon_distance,
            sample, forecast
        historical_observations : pd.DataFrame, optional
            Not used by this plot

        Returns
        -------
        ChartType
            Altair chart containing the complete evaluation dashboard
        """
        flat_observations = FlatObserved(observations)
        flat_forecasts = FlatForecasts(forecasts)

        charts = []

        # Title
        charts.append(
            alt.Chart()
            .mark_text(align="left", fontSize=20, fontWeight="bold")
            .encode(text=alt.value("Overview of Various Metrics Dashboard"))
            .properties(height=24)
        )

        # Explanatory text
        charts.append(
            text_chart(
                "This is a collection of various metrics plotted by forecast horizon and time period. "
                "Some of these might be useful for modelers or people trying to assess forecast quality over time or by horizon. ",
                line_length=80,
            )
        )

        plotting_classes = [MetricByHorizonV2Mean, MetricByTimePeriodV2Mean]
        for plotting_class in plotting_classes:
            metrics_to_show = [Coverage25_75Metric, RMSEMetric, CRPSNormMetric]
            for metric_factory in metrics_to_show:
                metric = metric_factory()
                name = metric.get_name()
                description = metric.get_description()
                title_plot = title_chart(name)
                charts.append(title_plot)
                textplot = text_chart(f"The metric shown below is '{name}'. Description: {description}", line_length=80)
                charts.append(textplot)
                metric_df = metric.get_detailed_metric(flat_observations, flat_forecasts)
                subplot = plotting_class(metric_df).plot(title=name)
                charts.append(subplot)

        # Combine all charts vertically
        dashboard = alt.vconcat(*charts).configure(
            axis={"labelFontSize": 11, "titleFontSize": 12},
            legend={"labelFontSize": 11, "titleFontSize": 12},
            view={"stroke": None},
        )

        return dashboard
