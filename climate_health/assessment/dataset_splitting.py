from typing import Iterable, Tuple, Protocol, Optional, Type

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthData, ClimateData, HealthData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import Year, Month
from climate_health.time_period.dataclasses import Period
from climate_health.time_period.relationships import previous
import dataclasses

def split_period_on_resolution(param, param1, resolution) -> Iterable[Month]:
    pass


def extend_to(period, future_length):
    pass


class IsTimeDelta(Protocol):
    pass


def split_test_train_on_period(data_set: IsSpatioTemporalDataSet, split_points: Iterable[Period],
                               future_length: Optional[IsTimeDelta] = None, include_future_weather: bool = False,
                               future_weather_class: Type[ClimateData] = ClimateData):
    func = train_test_split_with_weather if include_future_weather else train_test_split

    if include_future_weather:
        return (train_test_split_with_weather(data_set, period, future_length, future_weather_class) for period in split_points)
    return (func(data_set, period, future_length) for period in split_points)


def split_train_test_with_future_weather(data_set: IsSpatioTemporalDataSet, split_points: Iterable[Period],
                                         future_length: Optional[IsTimeDelta] = None):
    return (train_test_split(data_set, period, future_length) for period in split_points)


# Should we index on split-timestamp, first time period, or complete time?
def train_test_split(data_set: IsSpatioTemporalDataSet, prediction_start_period: Period,
                     extension: Optional[IsTimeDelta] = None):
    last_train_period = previous(prediction_start_period)
    train_data = data_set.restrict_time_period(slice(None, last_train_period))
    if extension is not None:
        end_period = prediction_start_period.extend_to(extension)
    else:
        end_period = None
    test_data = data_set.restrict_time_period(slice(prediction_start_period, end_period))

    return train_data, test_data


def train_test_split_with_weather(data_set: IsSpatioTemporalDataSet, prediction_start_period: Period,
                                  extension: Optional[IsTimeDelta] = None,
                                  future_weather_class: Type[ClimateData] = ClimateData):
    train_set, test_set = train_test_split(data_set, prediction_start_period, extension)
    tmp_values: Iterable[Tuple[str, ClimateHealthData]] = ((loc, temporal_data.data()) for loc, temporal_data in
                                                           test_set.items())
    future_weather = SpatioTemporalDict(
        {loc: future_weather_class(
            *[getattr(values, field.name) if hasattr(values, field.name) else values.mean_temperature for field in dataclasses.fields(future_weather_class)])
         for loc, values in tmp_values})
    return train_set, test_set, future_weather


def get_split_points_for_data_set(data_set: IsSpatioTemporalDataSet, max_splits: int, start_offset = 1) -> list[Period]:
    periods = next(iter(
        data_set.data())).data().time_period  # Uses the time for the first location, assumes it to be the same for all!
    return get_split_points_for_period_range(max_splits, periods, start_offset)


def get_split_points_for_period_range(max_splits, periods, start_offset):
    delta = (len(periods) - 1 - start_offset) // (max_splits+1)
    return list(periods)[start_offset+delta::delta][:max_splits]
