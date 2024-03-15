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
        plotting_data = pd.concat(results.values(), keys=results.keys(), names=['model']).reset_index()
        figs = cls._make_charts(plotting_data=plotting_data)
        return cls(figs)

    def save(self, filename: str):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'w') as f:
            for fig in self.report:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    @staticmethod
    def _make_charts(plotting_data: pd.DataFrame):
        across_time_periods_report = px.violin(plotting_data, x='period', y='mae', color='model', box=True,
                                               points="all", title="MAE across time periods")
        across_locations_report = px.violin(plotting_data, x='location', y='mae', color='model', box=True,
                                            points="all", title="MAE across locations")
        return across_time_periods_report, across_locations_report
