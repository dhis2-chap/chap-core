import numpy as np
import pandas as pd

from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet


class MultiRegionPredictor:
    def __init__(self):
        ...

    def _get_flat_data(self, data: DataSet[ClimateHealthTimeSeries]) -> pd.DataFrame:
        dfs = []
        for i, (location, location_data) in data.items():
            df = location_data.data().topandas()
            df['location'] = df['location']
            dfs.append(df)
        all_data = pd.concat(dfs)
        dfs['location'] = pd.Categorical(dfs['location'])
        return dfs

    def train(self, data: ClimateHealthTimeSeries):
        flat_data = self._get_flat_data(data)
        with pm.Model() as model:

