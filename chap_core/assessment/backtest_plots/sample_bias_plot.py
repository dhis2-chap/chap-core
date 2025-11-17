"""
Sample bias plot for backtests.

This module provides a BackTestPlotBase implementation that generates
evaluation dashboards showing the ratio of forecast samples above truth
by horizon distance and time period.
"""

import altair as alt

from chap_core.assessment.evaluation import Evaluation
from chap_core.database.tables import BackTest
from chap_core.plotting.backtest_plot import BackTestPlotBase, text_chart
from chap_core.assessment.flat_representations import FlatObserved, FlatForecasts
from chap_core.assessment.metrics.above_truth import RatioOfSamplesAboveTruth


class RatioOfSamplesAboveTruthBacktestPlot(BackTestPlotBase):
    """
    Backtest plot showing forecast bias relative to observations.

    This plot shows the ratio of forecast samples that are above the true
    observation value, organized by:
    - Horizon distance (how far ahead the forecast is)
    - Time period and location (when/where the observation occurred)
    """

    name = "Sample Bias Plot"
    description = "Backtest plot showing forecast bias relative to observations."

    def __init__(
        self,
        flat_observations: FlatObserved,
        flat_forecasts: FlatForecasts,
        title: str = "Sample Bias Dashboard",
    ):
        """
        Initialize the sample bias plot.

        Parameters
        ----------
        flat_observations : FlatObserved
            Observations in flat format
        flat_forecasts : FlatForecasts
            Forecasts in flat format
        title : str, optional
            Title for the dashboard
        """
        self._flat_observations = flat_observations
        self._flat_forecasts = flat_forecasts
        self._title = title

    @classmethod
    def from_backtest(
        cls, backtest: BackTest, title: str = "Sample Bias Dashboard"
    ) -> "RatioOfSamplesAboveTruthBacktestPlot":
        """
        Create a SampleBiasPlot from a BackTest object.

        Parameters
        ----------
        backtest : BackTest
            The backtest object containing forecast and observation data
        title : str, optional
            Title for the dashboard

        Returns
        -------
        RatioOfSamplesAboveTruthBacktestPlot
            An instance ready to generate the plot
        """
        # Use Evaluation abstraction to get flat representation
        evaluation = Evaluation.from_backtest(backtest)
        flat_data = evaluation.to_flat()

        return cls(flat_data.observations, flat_data.forecasts, title=title)

    def plot(self) -> alt.Chart:
        """
        Generate and return the dashboard visualization.

        Returns
        -------
        alt.Chart
            Altair chart containing the complete evaluation dashboard
        """
        # Compute the ratio of samples above truth metric
        metric = RatioOfSamplesAboveTruth()
        metric_df = metric.compute(self._flat_observations, self._flat_forecasts)

        charts = []

        # Title
        charts.append(
            alt.Chart()
            .mark_text(align="left", fontSize=20, fontWeight="bold")
            .encode(text=alt.value(self._title))
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
        # # Explanatory text
        # charts.append(
        #     alt.Chart()
        #     .mark_text(align="left", fontSize=12)
        #     .encode(
        #         text=alt.value(
        #             "These plots show biases in the samples returned by the model, "
        #             "\nspecifically whether the samples generally are over or under the true observation.\n"
        #             "This can be used to assess model calibration and tendency to over- or under-predict."
        #         )
        #     )
        #     .properties(height=18)
        # )

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

        return dashboard
