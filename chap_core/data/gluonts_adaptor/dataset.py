import dataclasses
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np
from gluonts.model import SampleForecast

from chap_core.assessment.dataset_splitting import train_test_split
from chap_core.file_io.example_data_set import datasets
from chap_core.datatypes import TimeSeriesData, remove_field, Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.spatio_temporal_data.multi_country_dataset import (
    MultiCountryDataSet,
)
from chap_core.time_period import PeriodRange
import logging

logger = logging.getLogger(__name__)
GlunTSDataSet = Iterable[dict]

T = TypeVar("T", bound=TimeSeriesData)


class ForecastAdaptor:
    @staticmethod
    def from_samples(samples: Samples) -> SampleForecast:
        start_period = samples.time_period[0].topandas()
        return SampleForecast(samples.samples.T, start_period)


class DataSetAdaptor:
    @staticmethod
    def _from_single_gluonts_series(series: dict, dataclass: type[T]) -> T:
        field_names = [
            field.name for field in dataclasses.fields(dataclass) if field.name not in ["disease_cases", "time_period"]
        ]
        field_dict = {name: series["feat_dynamic_real"].T[:, i] for i, name in enumerate(field_names)}
        field_dict["disease_cases"] = series["target"]
        field_dict["time_period"] = PeriodRange.from_start_and_n_periods(series["start"], len(series["target"]))
        return dataclass(**field_dict)

    @staticmethod
    def from_gluonts(gluonts_dataset: GlunTSDataSet, dataclass: type[T]) -> DataSet[T]:
        return DataSet(
            {
                series["feat_static_cat"][0]: DataSetAdaptor._from_single_gluonts_series(series, dataclass)
                for series in gluonts_dataset
            }
        )

    to_dataset = from_gluonts

    @staticmethod
    def get_metadata(dataset: DataSet):
        return {"static_cat": [{i: location for i, location in enumerate(dataset.keys())}]}

    @staticmethod
    def to_gluonts(dataset: DataSet, start_index=0, static=None, real=None) -> GlunTSDataSet:
        if isinstance(dataset, MultiCountryDataSet):
            yield from DataSetAdaptor.to_gluonts_multicountry(dataset)
            return
        static = [] if static is None else static
        assert real is None
        for i, (location, data) in enumerate(dataset.items(), start=start_index):
            period = data.time_period[0]

            yield {
                "start": period.topandas(),
                "target": data.disease_cases,
                "feat_dynamic_real": remove_field(data, "disease_cases").to_array().T,  # exclude the target
                "feat_static_cat": [i] + static,
            }

    from_dataset = to_gluonts

    @staticmethod
    def to_gluonts_testinstances(history: DataSet, future: DataSet, prediction_length):
        for i, (location, historic_data) in enumerate(history.items()):
            future_data = future[location]
            assert future_data.start_timestamp == historic_data.end_timestamp

            period = historic_data.time_period[0]
            historic_predictors = remove_field(historic_data, "disease_cases").to_array()

            future_predictors = future_data.to_array()
            logger.warning("Assuming location order is the same for test data")

            yield {
                "start": period.topandas(),
                "target": historic_data.disease_cases,
                "feat_dynamic_real": np.concatenate([historic_predictors, future_predictors], axis=0),
                "feat_static_cat": [i],
            }

    @staticmethod
    def to_gluonts_multicountry(dataset: MultiCountryDataSet) -> GlunTSDataSet:
        offset = 0
        for i, (country, data) in enumerate(dataset.items()):
            yield from DataSetAdaptor.to_gluonts(data, start_index=offset, static=[i])
            offset += len(data.keys())


def get_dataset(name, with_metadata=False):
    if name == "full":
        dataset = MultiCountryDataSet.from_folder(Path("/home/knut/Data/ch_data/full_data"))
        ds = DataSetAdaptor.to_gluonts(dataset)
    else:
        dataset = datasets[name].load()
        ds = DataSetAdaptor.to_gluonts(dataset)
    if with_metadata:
        return ds, DataSetAdaptor.get_metadata(dataset)
    return ds


def get_split_dataset(name, n_periods=6) -> tuple[GlunTSDataSet, GlunTSDataSet]:
    if name == "full":
        data_set = MultiCountryDataSet.from_folder(Path("/home/knut/Data/ch_data/full_data"))
    else:
        data_set = datasets[name].load()

    prediction_start = data_set.period_range[-n_periods]
    train, test = train_test_split(data_set, prediction_start, restrict_test=False)

    return DataSetAdaptor.to_gluonts(train), DataSetAdaptor.to_gluonts(test)
