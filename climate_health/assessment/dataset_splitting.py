from typing import Iterable, Tuple, Protocol, Optional

from climate_health.dataset import SpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthData, ClimateData, HealthData
from climate_health.time_period import Year, Month
from climate_health.time_period.dataclasses import Period
from climate_health.time_period.relationships import previous


def split_period_on_resolution(param, param1, resolution) -> Iterable[Month]:
    pass


def extend_to(period, future_length):
    pass


class TimeDelta(Protocol):
    pass


def split_test_train_on_period(data_set: SpatioTemporalDataSet, split_points: Iterable[Period],
                               future_length: TimeDelta):
    return (train_test_split(data_set, period, future_length) for period in split_points)


# Should we index on split-timestamp, first time period, or complete time?
def train_test_split(data_set: SpatioTemporalDataSet, prediction_start_period: Period,
                     extension: Optional[TimeDelta] = None):
    last_train_period = previous(prediction_start_period)
    train_data = data_set.restrict_time_period(slice(None, last_train_period))
    if extension is not None:
        end_period = prediction_start_period.extend_to(extension)
    else:
        end_period = None
    test_data = data_set.restrict_time_period(slice(prediction_start_period, end_period))

    return train_data, test_data
