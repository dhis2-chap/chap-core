from typing import Iterable, Tuple

from climate_health.dataset import SpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthData, ClimateData, HealthData
from climate_health.time_period import Year


def split_test_train_on_period(data_set: SpatioTemporalDataSet, resolution=Year) -> Iterable[Tuple[SpatioTemporalDataSet[ClimateHealthData], SpatioTemporalDataSet[ClimateData], SpatioTemporalDataSet[HealthData]]]:
    pass
