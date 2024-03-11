from typing import Iterable, Tuple, Protocol

from climate_health.dataset import SpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthData, ClimateData, HealthData
from climate_health.time_period import Year, Month
from climate_health.time_period.dataclasses import Period


def split_period_on_resolution(param, param1, resolution) -> Iterable[Month]:
    pass


def extend_to(period, future_length):
    pass


class TimeDelta(Protocol):
    pass


def split_test_train_on_period(data_set: SpatioTemporalDataSet, split_points: Iterable[Period], future_length: TimeDelta):
    return (test_train_split(data_set, period, future_length) for period in split_points)

# Should we index on split-timestamp, first time period, or complete time?
def test_train_split(data_set: SpatioTemporalDataSet, prediction_start_period: Period, extension: TimeDelta):
    train_data = data_set.restrict_time_period(end_period=prediction_start_period.previous())
    test_data = data_set.restrict_time_period(start_period=prediction_start_period,
                                              end_period=prediction_start_period.extend_to(extension))
    return train_data, test_data
