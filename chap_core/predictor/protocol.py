from typing import Protocol

from chap_core.data import DataSet
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

    def train(self, data: DataSet[ClimateHealthTimeSeries]): ...

    def sample(
        self, future_weather: DataSet[ClimateData], n_samples: int = 1
    ) -> DataSet[HealthData]: ...


class IsMultiRegionPredictor(Protocol):
    def train(
        self,
        spatio_temporal_climate_health_data: DataSet[ClimateHealthTimeSeries],
    ): ...

    def predict(self, future_weather: DataSet[ClimateData]) -> DataSet[HealthData]: ...
