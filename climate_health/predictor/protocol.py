from typing import Protocol
from ..datatypes import ClimateData, ClimateHealthTimeSeries, HealthData


class Predictor(Protocol):
    def __init__(self):
        pass

    def predict(self, x):
        pass

    def train(self, x, y):
        pass


class Sampler(Protocol):
    '''
    Model that can sample forward in time given a set of weather data.
    '''

    def train(self, time_series: ClimateHealthTimeSeries):
        ...

    def sample(self, weather_data: ClimateData, n_samples: int = 1) -> HealthData:
        ...
