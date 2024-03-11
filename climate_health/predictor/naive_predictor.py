import numpy as np

from climate_health.dataset import SpatioTemporalDataSet
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.datatypes import HealthData, ClimateHealthTimeSeries, ClimateData
from climate_health.time_period.dataclasses import Period


class NaivePredictor:
    ''' This should be a linear regression of prev cases and season'''

    def __init__(self, lead_time=1):
        self._average_cases = None

    def train(self, data: ClimateHealthTimeSeries):
        self._average_cases = data.disease_cases.mean()

    def predict(self, future_climate_data: ClimateData) -> HealthData:
        return HealthData(future_climate_data.time_period, np.full(len(future_climate_data), self._average_cases))




class MultiRegionNaivePredictor:
    '''TODO: This should be a linear regression of prev cases and season for each location.'''

    def __init__(self, lead_time=1, resolution=1):
        self._training_stop = None
        self._average_cases = None
        self._resolution = resolution

    def train(self, data: SpatioTemporalDataSet[ClimateHealthTimeSeries]):
        self._average_cases = {location: data.disease_cases.mean() for location, data in data.items()}
        self._training_stop = data.stop_time()

    def predict(self, future_weather: SpatioTemporalDataSet[ClimateData]) -> HealthData:

        prediction_dict = {HealthData(self.prediction_period, np.full(len(future_weather[location]), self._average_cases[location])) for
                           location in future_weather.keys()}
        return SpatioTemporalDict(prediction_dict)

    @property
    def prediction_period(self):
        return Period(self._training_stop+self._lead_time, self._resolution)

class NaiveForecastSampler:
    def __init__(self):
        self._case_average = None
        self._case_std = None

    def train(self, time_series: ClimateHealthTimeSeries):
        self._case_average = time_series.disease_cases.mean()
        self._case_std = time_series.disease_cases.std()

    def sample(self, weather_data: ClimateData, n_samples: int = 1) -> HealthData:
        return HealthData(weather_data.time_period, np.random.normal(self._case_average, self._case_std, n_samples))
