import altair as alt

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import DetailedRMSE, DetailedCRPSNorm, IsWithin25th75thDetailed
from chap_core.database.tables import BackTest
from chap_core.plotting.backtest_plot import BackTestPlotBase, text_chart, title_chart
from chap_core.assessment.flat_representations import FlatObserved, FlatForecasts
from chap_core.plotting.evaluation_plot import MetricByHorizonV2Mean, MetricByTimePeriodV2Mean


# Import your metrics here
# from chap_core.assessment.metrics.your_metric import YourMetric


class BackTestPlot1(BackTestPlotBase):
    """
    Backtest plot (as prototype/example) with various metrics
    """

    name = "Overview of various metrics by horizon/time"
    description = "A dashboard showing various metrics by forecast horizon and time period."

    def __init__(
        self,
        flat_observations: FlatObserved,
        flat_forecasts: FlatForecasts,
        title: str = "Overview of Various Metrics Dashboard",
    ):
        """
        Initialize the backtest plot.

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
    def from_backtest(cls, backtest: BackTest, title: str = "BackTest Dashboard 1") -> "BackTestPlot1":
        """
        Create a BackTestPlot1 from a BackTest object.

        Parameters
        ----------
        backtest : BackTest
            The backtest object containing forecast and observation data
        title : str, optional
            Title for the dashboard

        Returns
        -------
        BackTestPlot1
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
        # metric = YourMetric()
        # metric_df = metric.compute(self._flat_observations, self._flat_forecasts)

        charts = []

        # Title
        charts.append(
            alt.Chart()
            .mark_text(align="left", fontSize=20, fontWeight="bold")
            .encode(text=alt.value(self._title))
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
            metrics_to_show = [IsWithin25th75thDetailed, DetailedRMSE, DetailedCRPSNorm]
            for metric in metrics_to_show:
                name = metric().spec.metric_name
                description = metric().spec.description
                title_plot = title_chart(name)
                charts.append(title_plot)
                textplot = text_chart(f"The metric shown below is '{name}'. Description: {description}", line_length=80)
                charts.append(textplot)
                metric_df = metric().get_metric(self._flat_observations, self._flat_forecasts)
                print("MEtric df")
                print(metric_df)
                subplot = plotting_class(metric_df).plot(title=name)
                charts.append(subplot)

        #

        # Combine all charts vertically
        dashboard = alt.vconcat(*charts).configure(
            axis={"labelFontSize": 11, "titleFontSize": 12},
            legend={"labelFontSize": 11, "titleFontSize": 12},
            view={"stroke": None},
        )

        return dashboard
