"""
Sample bias plot for backtests.

This module provides a backtest plot that shows the ratio of forecast samples
above truth by horizon distance and time period.
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import ChartType, FacetDimension, FacetedBacktestPlot, backtest_plot
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.assessment.metrics import RatioAboveTruthMetric
from chap_core.plotting.backtest_plot import text_chart


@backtest_plot(
    plot_id="ratio_of_samples_above_truth",
    name="Sample Bias Plot",
    description="Backtest plot showing forecast bias relative to observations.",
)
class SampleBiasPlot(FacetedBacktestPlot):
    """
    Backtest plot showing forecast bias relative to observations.

    This plot shows the ratio of forecast samples that are above the true
    observation value, organized by:
    - Horizon distance (how far ahead the forecast is)
    - Time period and location (when/where the observation occurred)

    It is an aggregate dashboard with no per-coordinate decomposition, so it
    declares no facet dimensions; it conforms to the faceting interface only so
    every registered plot shares the same shape.
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

    def _plot(self, metric_df: pd.DataFrame) -> ChartType:
        """Render the bias dashboard from the metric frame."""
        charts = []

        # Title
        charts.append(
            alt.Chart()
            .mark_text(align="left", fontSize=20, fontWeight="bold")
            .encode(text=alt.value("Sample Bias Dashboard"))
            .properties(height=24)
        )

        charts.append(
            text_chart(
                "These plots show biases in the samples returned by the model, "
                "\nspecifically whether the samples generally are over or under the true observation.\n"
                "This can be used to assess model calibration and tendency to over- or under-predict.",
                line_length=50,
            )
        )

        # By horizon: aggregate by horizon_distance (mean across locations and time periods)
        horizon_df = metric_df.groupby(["horizon_distance"]).agg({"metric": "mean"}).reset_index()
        horizon_chart = (
            alt.Chart(horizon_df)
            .mark_bar(point=True)
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("metric:Q", title="Ratio of samples above truth"),
                tooltip=["horizon_distance", "metric"],
            )
            .properties(
                width=600,
                height=400,
                title={
                    "text": "Ratio of samples above truth by horizon",
                    "subtitle": "Ratio (0.0-1.0) of forecast samples > truth. x-axis: forecast horizon distance",
                    "anchor": "start",
                    "fontSize": 16,
                    "subtitleFontSize": 12,
                },
            )
        )
        charts.append(horizon_chart)

        # By time period and location: aggregate by time_period and location (mean across horizons)
        time_df = metric_df.groupby(["time_period", "location"]).agg({"metric": "mean"}).reset_index()
        time_chart = (
            alt.Chart(time_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_period:O", title="Time period"),
                y=alt.Y("metric:Q", title="Ratio of samples above truth"),
                color=alt.Color("location:N", title="Location"),
                tooltip=["time_period", "location", "metric"],
            )
            .properties(
                width=600,
                height=400,
                title={
                    "text": "Ratio of samples above truth by time period",
                    "subtitle": "Ratio (0.0-1.0) of forecast samples > truth. x-axis: time period of observation",
                    "anchor": "start",
                    "fontSize": 16,
                    "subtitleFontSize": 12,
                },
            )
        )
        charts.append(time_chart)

        # Combine all charts vertically
        dashboard = alt.vconcat(*charts).configure(
            axis={"labelFontSize": 11, "titleFontSize": 12},
            legend={"labelFontSize": 11, "titleFontSize": 12},
            view={"stroke": None},
        )

        return dashboard  # type: ignore[no-any-return]
