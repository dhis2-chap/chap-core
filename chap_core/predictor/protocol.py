from typing import Protocol

from ..datatypes import ClimateData, ClimateHealthTimeSeries, HealthData


class IsSampler(Protocol):
    """
    Model that can sample forward in time given a set of weather data.
    """

    def train(self, time_series: ClimateHealthTimeSeries): ...

    def sample(self, weather_data: ClimateData, n_samples: int = 1) -> HealthData: ...
