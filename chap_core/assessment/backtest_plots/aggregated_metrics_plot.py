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
            # Create a table showing the metrics
            table_data = metrics_df[["metric_name", "metric_id", "value"]].copy()
            table_data.columns = ["Metric Name", "Metric ID", "Value"]
            table_data["Value"] = table_data["Value"].apply(lambda x: f"{x:.6f}")

            table = (
                alt.Chart(table_data)
                .mark_text(align="left", baseline="middle")
                .encode(
                    x=alt.X("column:N", axis=alt.Axis(title=None, labelAngle=0, labelFontSize=12)),
                    y=alt.Y("row:O", axis=None),
                    text=alt.Text("value:N"),
                )
                .transform_window(row="row_number()")
                .transform_fold(["Metric Name", "Metric ID", "Value"], as_=["column", "value"])
                .properties(width=600, height=max(150, len(metrics_df) * 25 + 30))
            )
            charts.append(table)

        # Combine all charts vertically
        dashboard = alt.vconcat(*charts).configure(
            axis={"labelFontSize": 11, "titleFontSize": 12},
            legend={"labelFontSize": 11, "titleFontSize": 12},
            view={"stroke": None},
        )

        return dashboard
