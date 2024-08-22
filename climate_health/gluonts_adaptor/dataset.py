from typing import Iterable

from ..assessment.dataset_splitting import train_test_split
from ..file_io.example_data_set import datasets
from climate_health.datatypes import TimeSeriesData, remove_field
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from ..time_period import delta_month

GlunTSDataSet = Iterable[dict]


class DataSetAdaptor:
    @staticmethod
    def from_gluonts(self, gluonts_dataset: GlunTSDataSet, dataclass: type[TimeSeriesData]) -> SpatioTemporalDict:
        raise NotImplementedError

    @staticmethod
    def to_gluonts(dataset: SpatioTemporalDict) -> GlunTSDataSet:
        n_locations = len(dataset.keys())
        for i, (location, data) in enumerate(dataset.items()):
            period = data.time_period[0]
            yield {
                'start': period.topandas(),
                'target': data.disease_cases,
                'feat_dynamic_real': remove_field(data, 'disease_cases').to_array(),  # exclude the target
                'feat_static_cat': [i],
            }


def get_dataset(name):
    return DataSetAdaptor.to_gluonts(datasets[name].load())

def get_split_dataset(name, n_periods=6) -> tuple[GlunTSDataSet, GlunTSDataSet]:
    data_set = datasets[name].load()
    prediction_start = data_set.period_range[-n_periods]
    train, test = train_test_split(data_set, prediction_start, restrict_test=False)
    return DataSetAdaptor.to_gluonts(train), DataSetAdaptor.to_gluonts(test)