from typing import Iterable

from climate_health.datatypes import TimeSeriesData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict

GlunTSDataSet = Iterable[dict]

class DataSetAdaptor:
    @staticmethod
    def from_gluonts(self, gluonts_dataset: GlunTSDataSet, dataclass: type[TimeSeriesData]) -> SpatioTemporalDict:
        raise NotImplementedError

    @staticmethod
    def to_gluonts(dataset: SpatioTemporalDict) -> GlunTSDataSet:
        n_locations=len(dataset.keys())
        for i, (location, data) in enumerate(dataset.items()):

            period = data.time_period[0]
            print(dir(period))
            yield {
                'start': period.topandas(),
                'target': data.disease_cases,
                'feat_dynamic_real': data.mean_temperature,
                'feat_static_cat': [i],
            }