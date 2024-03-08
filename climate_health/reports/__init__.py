import os.path
import pandas as pd
import plotly.express as px
from ..datatypes import ResultType


class HTMLReport:
    def __init__(self, report):
        self.report = report
    @classmethod
    def from_results(cls, results: dict[str, ResultType]) -> 'HTMLReport':
        plotting_data = pd.concat(results.values(), keys=results.keys(), names=['model']).reset_index()
        fig = px.line(plotting_data, x='period', y='mae', color='model', facet_row='location', line_group='model')
        return cls(fig)

    def save(self, filename: str):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.report.write_html(filename)