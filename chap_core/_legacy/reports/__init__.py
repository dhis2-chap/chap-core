import os.path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chap_core.datatypes import ResultType


class HTMLReport:
    """A class to generate and save an HTML report from a dictionary of results."""

    error_measure = "mae"

    def __init__(self, report: tuple[go.Figure]):
        self.report = tuple(report)

    @classmethod
    def from_results(cls, results: dict[str, ResultType]) -> "HTMLReport":
        table = pd.concat(results.values(), keys=results.keys(), names=["model"]).reset_index()
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> "HTMLReport":
        plotting_data = cls._prepare_plotting_data(table)
        figs = cls._make_charts(plotting_data=plotting_data)
        return cls(figs)

    @classmethod
    def _prepare_plotting_data(cls, plotting_data) -> pd.DataFrame:
        plotting_data["period"] = pd.to_datetime(plotting_data["period"])
        plotting_data["month"] = plotting_data["period"].dt.month
        plotting_data["week"] = plotting_data["period"].dt.isocalendar().week
        return plotting_data

    def save(self, filename: str):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "w") as f:
            for fig in self.report:
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    @staticmethod
    def _make_charts(plotting_data: pd.DataFrame):
        for category in ["period", "location", "month", "week"]:
            yield HTMLReport._make_violin_plot(plotting_data, category)

    @classmethod
    def _make_violin_plot(cls, plotting_data: pd.DataFrame, category: str) -> go.Figure:
        return px.violin(
            plotting_data,
            x=category,
            y=cls.error_measure,
            color="model",
            box=True,
            points="all",
            title=f"{cls.error_measure.capitalize()} across {category}",
        )


class HTMLSummaryReport(HTMLReport):
    @staticmethod
    def _tidy_summary_table(table: pd.DataFrame) -> pd.DataFrame:
        """Make one row each for mean, median, std, min, max, quantile_low, quantile_high"""
        return table.melt(
            id_vars=["location", "period", "mae", "mle", "model"],
            var_name="statistic",
            value_name="value",
        )

    @classmethod
    def _prepare_plotting_data(cls, plotting_data) -> pd.DataFrame:
        plotting_data["period"] = pd.to_datetime(plotting_data["period"])
        return plotting_data
        # return cls._tidy_summary_table(plotting_data)

    @classmethod
    def _make_charts(cls, plotting_data: pd.DataFrame):
        # mask = plotting_data['statistic'].isin(['quantile_low', 'quantile_high', 'true_value'])
        plotting_data["error_minus"] = plotting_data["median"] - plotting_data["quantile_low"]
        plotting_data["error_plus"] = plotting_data["quantile_high"] - plotting_data["median"]
        plotting_data["median_p1"] = plotting_data["median"] + 1
        yield px.scatter(
            plotting_data,
            x="location",
            y="median_p1",
            error_y_minus="error_minus",
            error_y="error_plus",
            color="model",
            facet_row="period",
            title="Summary statistics",
            log_y=True,
        )


class HTMLForecastReport:
    """
    Take in dataframe with multiple observations per spilt point and
    make a forecast report with forecasted values from multiple points in one plot
    """

    ...
