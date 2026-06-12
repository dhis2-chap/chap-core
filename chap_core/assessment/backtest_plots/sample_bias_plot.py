"""
Sample bias plots for backtests.

This module provides backtest plots that show the ratio of forecast samples
above truth, one by horizon distance and one by time period.
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import ChartType, FacetDimension, FacetedBacktestPlot, backtest_plot
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.assessment.metrics import RatioAboveTruthMetric

_BIAS_DESCRIPTION = (
    "Shows biases in the samples returned by the model, specifically whether the samples "
    "generally are over or under the true observation. This can be used to assess model "
    "calibration and tendency to over- or under-predict."
)


class SampleBiasPlotBase(FacetedBacktestPlot):
    """
    Shared preprocessing for the sample bias plots.

    Computes the ratio of forecast samples that are above the true observation
    value per forecast row. The plots are aggregates with no per-coordinate
    decomposition, so they declare no facet dimensions; they conform to the
    faceting interface only so every registered plot shares the same shape.
    """

    facet_dimensions: list[FacetDimension] = []

    def _preprocess(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute the ratio-of-samples-above-truth metric per forecast row."""
        flat_observations = FlatObserved(observations)
        flat_forecasts = FlatForecasts(forecasts)
        metric = RatioAboveTruthMetric()
        return metric.get_detailed_metric(flat_observations, flat_forecasts)


@backtest_plot(
    plot_id="sample_bias_by_horizon",
    name="Sample Bias by Horizon",
    description=f"Ratio of forecast samples above truth per forecast horizon. {_BIAS_DESCRIPTION}",
)
class SampleBiasByHorizonPlot(SampleBiasPlotBase):
    """Bar chart of the mean ratio of samples above truth per forecast horizon."""

    def _plot(self, metric_df: pd.DataFrame) -> ChartType:
        horizon_df = metric_df.groupby(["horizon_distance"]).agg({"metric": "mean"}).reset_index()
        return (  # type: ignore[no-any-return]
            alt.Chart(horizon_df)
            .mark_bar()
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("metric:Q", title="Ratio of samples above truth"),
                tooltip=["horizon_distance", "metric"],
            )
            .properties(height=300)
        )


@backtest_plot(
    plot_id="sample_bias_by_time_period",
    name="Sample Bias by Time Period",
    description=(
        f"Ratio of forecast samples above truth over time periods, one line per location. {_BIAS_DESCRIPTION}"
    ),
)
class SampleBiasByTimePeriodPlot(SampleBiasPlotBase):
    """Line chart of the mean ratio of samples above truth per time period and location."""

    def _plot(self, metric_df: pd.DataFrame) -> ChartType:
        time_df = metric_df.groupby(["time_period", "location"]).agg({"metric": "mean"}).reset_index()
        return (  # type: ignore[no-any-return]
            alt.Chart(time_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_period:O", title="Time period"),
                y=alt.Y("metric:Q", title="Ratio of samples above truth"),
                color=alt.Color("location:N", title="Location"),
                tooltip=["time_period", "location", "metric"],
            )
            .properties(height=300)
        )
