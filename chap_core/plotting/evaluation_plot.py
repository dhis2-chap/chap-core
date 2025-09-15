import abc
from typing import Optional

import altair as alt
import pandas as pd
from chap_core.assessment.metric_table import create_metric_table
from chap_core.database.base_tables import DBModel
from chap_core.metrics.metrics import Metric
alt.renderers.enable('browser')


class MetricPlot(abc.ABC):
    def __init__(self, metrics: list[Metric], geojson: Optional[dict] = None):
        self._metrics = metrics
        self._geojson = geojson

    @abc.abstractmethod
    def plot(self) -> alt.Chart:
        pass

class VisualizationInfo(DBModel):
    id: str
    display_name: str
    description: str


class MetricByHorizon(MetricPlot):
    visualization_info = VisualizationInfo(
        id='metric_by_horizon',
        display_name="Horizon Plot",
        description='Shows the aggregated metric by forecast horizon')


    def plot_from_df(self, df: pd.DataFrame) -> alt.Chart:
        # aggregate for each horizon
        adf  = df.groupby(['horizon', 'org_unit']).agg({'value': 'mean'}).reset_index()
        chart = alt.Chart(adf).mark_bar(point=True).encode(
            x=alt.X('horizon:O', title='Horizon (periods ahead)'),
            y=alt.Y('value:Q', title='Mean Metric Value'),
            tooltip=['horizon', 'org_unit', 'value']
        ).properties(
            width=600,
            height=400,
            title='Mean Metric by Horizon'
        ).interactive()

        return chart

    def plot(self) -> alt.Chart:
        return self.plot_from_df(create_metric_table(self._metrics))

    def plot_spec(self) -> dict:
        chart = self.plot()
        return chart.to_dict()

class MetricMap(MetricPlot):
    visualization_info = VisualizationInfo(
        id='metric_map',
        display_name="Map",
        description='Shows a map of aggregated metrics per org unit'
    )

    def plot_from_df(self, df: pd.DataFrame) -> alt.Chart:
        # 2. Example values per region
        #data = pd.DataFrame({"region_id": [1, 2, 3, 4], "value": [10, 50, 30, 70]})
        data = df
        # 3. Convert GeoDataFrame to JSON (FeatureCollection)
        geojson_data = self._geojson

        # 4. Build Altair chart
        #    Use feature properties to join with your values DataFrame
        chart = (
            alt.Chart(alt.Data(values=geojson_data["features"]))
            .mark_geoshape(stroke="black", strokeWidth=0.5)
            .encode(
                color=alt.Color("value:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["properties.:N", "value:Q"],
            )
            .transform_lookup(lookup="properties.region_id", from_=alt.LookupData(data, "region_id", ["value"]))
            .project(type="identity")  # no reprojection; assumes coords already in lon/lat
            .properties(width=600, height=400)
        )

        return chart

    def plot(self) -> alt.Chart:
        return self.plot_from_df(create_metric_table(self._metrics))

    def plot_spec(self) -> dict:
        chart = self.plot()
        return chart.to_dict()
