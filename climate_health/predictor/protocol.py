from typing import Protocol

from .._legacy_dataset import IsSpatioTemporalDataSet
from ..datatypes import ClimateData, ClimateHealthTimeSeries, HealthData


class IsPredictor(Protocol):
    def __init__(self):
        pass

    def predict(self, x):
        pass

    def train(self, x, y):
        pass


class IsSampler(Protocol):
    """
    Model that can sample forward in time given a set of weather data.
    """

    def train(self, time_series: ClimateHealthTimeSeries): ...

    def sample(self, weather_data: ClimateData, n_samples: int = 1) -> HealthData: ...


class IsMultiRegionForecastSampler(Protocol):
    """
    Model that can sample forward for multiple locations in time given a set of weather data.
    """

    def train(self, data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries]): ...

    def sample(
        self, future_weather: IsSpatioTemporalDataSet[ClimateData], n_samples: int = 1
    ) -> IsSpatioTemporalDataSet[HealthData]: ...


class IsMultiRegionPredictor(Protocol):
    def train(
        self,
        spatio_temporal_climate_health_data: IsSpatioTemporalDataSet[
            ClimateHealthTimeSeries
        ],
    ): ...

    def predict(
        self, future_weather: IsSpatioTemporalDataSet[ClimateData]
    ) -> IsSpatioTemporalDataSet[HealthData]: ...
