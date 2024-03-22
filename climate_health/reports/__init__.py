import os.path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..datatypes import ResultType


class HTMLReport:
    """A class to generate and save an HTML report from a dictionary of results."""
    def __init__(self, report: tuple[go.Figure]):
        self.report = report

    @classmethod
    def from_results(cls, results: dict[str, ResultType]) -> 'HTMLReport':
        plotting_data = cls._prepare_plotting_data(results)
        figs = cls._make_charts(plotting_data=plotting_data)
        return cls(figs)

    @classmethod
    def _prepare_plotting_data(cls, results: dict[str, ResultType]) -> pd.DataFrame:
        plotting_data = pd.concat(results.values(), keys=results.keys(), names=['model']).reset_index()
        plotting_data["period"] = pd.to_datetime(plotting_data["period"])
        plotting_data["month"] = plotting_data["period"].dt.month
        plotting_data["week"] = plotting_data["period"].dt.isocalendar().week
        return plotting_data

    def save(self, filename: str):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'w') as f:
            for fig in self.report:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    @staticmethod
    def _make_charts(plotting_data: pd.DataFrame):
        for category in ['period', 'location', 'month', 'week']:
            yield HTMLReport._make_violin_plot(plotting_data, category)
        
    @classmethod
    def _make_violin_plot(cls, plotting_data: pd.DataFrame, category: str) -> go.Figure:
        return px.violin(plotting_data, x=category, y='mae', color='model', box=True, points="all",
                         title=f"MAE across {category}")
