import altair as alt
import pandas as pd

from chap_core.assessment.metrics import available_metrics
from chap_core.database.tables import BackTest
from chap_core.plotting.backtest_plot import BackTestPlotBase, text_chart, title_chart
from chap_core.assessment.flat_representations import (
    FlatObserved,
    FlatForecasts,
    convert_backtest_to_flat_forecasts,
    convert_backtest_observations_to_flat_observations,
)


class AggregatedMetricsPlot(BackTestPlotBase):
    """
    Backtest plot showing aggregated metrics (global metrics with no dimensions).
    """

    name = "Aggregated Metrics Overview"
    description = "A plot showing all global aggregated metrics from the assessment module."

    def __init__(
        self,
        flat_observations: FlatObserved,
        flat_forecasts: FlatForecasts,
        title: str = "Aggregated Metrics Overview",
    ):
        """
        Initialize the aggregated metrics plot.

        Parameters
        ----------
        flat_observations : FlatObserved
            Observations in flat format
        flat_forecasts : FlatForecasts
            Forecasts in flat format
        title : str, optional
            Title for the plot
        """
        self._flat_observations = flat_observations
        self._flat_forecasts = flat_forecasts
        self._title = title

    @classmethod
    def from_backtest(cls, backtest: BackTest, title: str = "Aggregated Metrics") -> "AggregatedMetricsPlot":
        """
        Create an AggregatedMetricsPlot from a BackTest object.

        Parameters
        ----------
        backtest : BackTest
            The backtest object containing forecast and observation data
        title : str, optional
            Title for the plot

        Returns
        -------
        AggregatedMetricsPlot
            An instance ready to generate the plot
        """
        flat_forecasts = FlatForecasts(convert_backtest_to_flat_forecasts(backtest.forecasts))
        flat_observations = FlatObserved(
            convert_backtest_observations_to_flat_observations(backtest.dataset.observations)
        )

        return cls(flat_observations, flat_forecasts, title=title)

    def plot(self) -> alt.Chart:
        """
        Generate and return the aggregated metrics visualization.

        Returns
        -------
        alt.Chart
            Altair chart containing the aggregated metrics
        """
        # Get all global metrics (metrics with no output dimensions)
        global_metrics = {
            metric_id: metric_cls
            for metric_id, metric_cls in available_metrics.items()
            if metric_cls().is_full_aggregate()
        }

        # Compute all global metrics
        metrics_data = []
        for metric_id, metric_cls in global_metrics.items():
            metric = metric_cls()
            metric_df = metric.get_metric(self._flat_observations, self._flat_forecasts)
            if len(metric_df) == 1:
                metric_value = float(metric_df["metric"].iloc[0])
                metrics_data.append(
                    {
                        "metric_id": metric_id,
                        "metric_name": metric.spec.metric_name,
                        "value": metric_value,
                        "description": metric.spec.description,
                    }
                )

        # Create DataFrame with metrics
        metrics_df = pd.DataFrame(metrics_data)

        charts = []

        # Title
        charts.append(title_chart(self._title))

        # Explanatory text
        charts.append(
            text_chart(
                "This plot shows all aggregated (global) metrics computed across all locations, "
                "time periods, and forecast horizons. These metrics provide a single summary "
                "value for the entire backtest evaluation.",
                line_length=80,
            )
        )

        if len(metrics_df) == 0:
            charts.append(
                text_chart(
                    "No global aggregated metrics found. All available metrics produce per-location, "
                    "per-time, or per-horizon breakdowns.",
                    line_length=80,
                )
            )
        else:
            # Create a bar chart of metrics
            bar_chart = (
                alt.Chart(metrics_df)
                .mark_bar()
                .encode(
                    x=alt.X("value:Q", title="Metric Value"),
                    y=alt.Y("metric_name:N", title="Metric", sort="-x"),
                    color=alt.Color("metric_name:N", legend=None),
                    tooltip=["metric_name:N", "value:Q", "description:N"],
                )
                .properties(width=600, height=max(200, len(metrics_df) * 30), title="Global Metric Values")
            )
            charts.append(bar_chart)

            # Add a text chart with metric details
            charts.append(title_chart("Metric Details", font_size=18))
            for _, row in metrics_df.iterrows():
                detail_text = f"{row['metric_name']} ({row['metric_id']}): {row['value']:.4f}"
                if row["description"] and row["description"] != "No description provided":
                    detail_text += f" - {row['description']}"
                charts.append(text_chart(detail_text, line_length=80, font_size=11))

        # Combine all charts vertically
        dashboard = alt.vconcat(*charts).configure(
            axis={"labelFontSize": 11, "titleFontSize": 12},
            legend={"labelFontSize": 11, "titleFontSize": 12},
            view={"stroke": None},
        )

        return dashboard
