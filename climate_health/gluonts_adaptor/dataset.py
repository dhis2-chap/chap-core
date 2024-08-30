from pathlib import Path
from typing import Iterable
from ..assessment.dataset_splitting import train_test_split
from ..file_io.example_data_set import datasets
from climate_health.datatypes import TimeSeriesData, remove_field
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from ..spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from ..time_period import delta_month

GlunTSDataSet = Iterable[dict]


class DataSetAdaptor:
    @staticmethod
    def from_gluonts(self, gluonts_dataset: GlunTSDataSet, dataclass: type[TimeSeriesData]) -> SpatioTemporalDict:
        raise NotImplementedError

    @staticmethod
    def to_gluonts(dataset: SpatioTemporalDict, start_index=0, static=None, real=None) -> GlunTSDataSet:
        if isinstance(dataset, MultiCountryDataSet):
            yield from DataSetAdaptor.to_gluonts_multicountry(dataset)
            return
        static = [] if static is None else static
        assert real is None
        for i, (location, data) in enumerate(dataset.items(), start=start_index):
            period = data.time_period[0]
            yield {
                'start': period.topandas(),
                'target': data.disease_cases,
                'feat_dynamic_real': remove_field(data, 'disease_cases').to_array(),  # exclude the target
                'feat_static_cat': [i]+static,
            }

    @staticmethod
    def to_gluonts_multicountry(dataset: MultiCountryDataSet) -> GlunTSDataSet:
        offset = 0
        for i, (country, data) in enumerate(dataset.items()):
            yield from DataSetAdaptor.to_gluonts(data, start_index=offset, static=[i])
            offset += len(data.keys())


def get_dataset(name):
    if name == 'full':
        data_set = MultiCountryDataSet.from_folder(Path('/home/knut/Data/ch_data/full_data'))
        return DataSetAdaptor.to_gluonts(data_set)
    return DataSetAdaptor.to_gluonts(datasets[name].load())

def get_split_dataset(name, n_periods=6) -> tuple[GlunTSDataSet, GlunTSDataSet]:
    if name == 'full':
        data_set = MultiCountryDataSet.from_folder(Path('/home/knut/Data/ch_data/full_data'))
    else:
        data_set = datasets[name].load()
    prediction_start = data_set.period_range[-n_periods]
    train, test = train_test_split(data_set, prediction_start, restrict_test=False)
    return DataSetAdaptor.to_gluonts(train), DataSetAdaptor.to_gluonts(test)