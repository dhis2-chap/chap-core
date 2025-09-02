import abc
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import pydantic
alt.renderers.enable('browser')
from chap_core.assessment.metric_table import create_metric_table


class Metric(abc.ABC):
    @abc.abstractmethod
    def compute(self, truth: float, predictions: np.ndarray) -> float:
        ...

    def aggregate(self, values: list[float]) -> float:
        return float(np.mean(values))

    def finalize(self, value: float) -> float:
        return value

class CRPS(Metric):
    def compute(self, truth, predictions) -> float:
        term1 = np.mean(np.abs(predictions - truth))
        term2 = 0.5 * np.mean(np.abs(predictions[:, None] - predictions[None, :]))
        return float(term1 - term2)

class AggregationInfo(pydantic.BaseModel):
    '''
    This class describes whether a metric is aggregated over temporal or spatial dimensions before being put in the database.
    '''
    temporal: bool = False
    spatial: bool = False
    samples: bool = True

def get_aggreagtion_level(metric_id: str):
    return AggregationInfo(temporal=False, spatial=False)


class MetricRegistry:
    metrics = ['crps_mean',
               'crps_norm_mean',
               'ratio_within_10th_90th',
               'ratio_within_90th']
    seeded_aggregations = {name: np.mean for name in metrics}

    def get_aggregation_function(self, metric_id: str):
        # Placeholder for actual implementation
        return self.seeded_aggregations.get(metric_id)

class MetricPlot(abc.ABC):
    def __init__(self, metrics: list[Metric], geojson: Optional[dict] = None):
        self._metrics = metrics
        self._geojson = geojson

    @abc.abstractmethod
    def plot(self) -> alt.Chart:
        pass


class MetricByHorizon(MetricPlot):
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
