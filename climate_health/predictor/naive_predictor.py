import numpy as np
import sklearn

from climate_health.dataset import IsSpatioTemporalDataSet
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

    def __init__(self, *args, **kwargs):
        self._training_stop = None
        self._average_cases = None

    def _get_mean(self, data):
        y = data.data().disease_cases
        y = y[~np.isnan(y)]
        return y.mean()
        # return data.data().disease_cases.mean()

    def train(self, data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries]):
        self._average_cases = {location: self._get_mean(data) for location, data in data.items()}
        #self._buffer = next(iter(data.values())).time_period[-1]

    def predict(self, future_weather: IsSpatioTemporalDataSet[ClimateData]) -> HealthData:
        prediction_dict = {location: HealthData(entry.data().time_period[:1], np.full(1, self._average_cases[location])) for
                           location, entry in future_weather.items()}
        return SpatioTemporalDict(prediction_dict)


class MultiRegionPoissonModel:
    def __init__(self, *args, **kwargs):
        self._training_stop = None
        self._models = {}
        self._saved_state = {}

    def _create_feature_matrix(self, data: ClimateHealthTimeSeries):
        lagged_values = data.disease_cases[:-1]
        season = data.time_period.month[1:, None] == np.arange(1, 13)
        return np.hstack([lagged_values, season])

    def train(self, data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries]):

        for location, location_data in data.items():
            X = self._create_feature_matrix(location_data)
            y = location_data.disease_cases[1:]

            model = sklearn.linear_model.PoissonRegressor()
            model.fit(X, y)
            self._models[location] = model
            self._saved_state[location] = location_data[-1:]

    def predict(self, data: IsSpatioTemporalDataSet[ClimateData]) -> IsSpatioTemporalDataSet[HealthData]:
        prediction_dict = {}
        for location, location_data in data.items():
            X = self._create_feature_matrix(np.concatenate([self._saved_state[location], location_data]))
            prediction = self._models[location].predict(X)
            prediction_dict[location] = HealthData(location_data.time_period[:1], np.atleast_1d(prediction))
        return SpatioTemporalDict(prediction_dict)


class NaiveForecastSampler:
    def __init__(self):
        self._case_average = None
        self._case_std = None

    def train(self, time_series: ClimateHealthTimeSeries):
        self._case_average = time_series.disease_cases.mean()
        self._case_std = time_series.disease_cases.std()

    def sample(self, weather_data: ClimateData, n_samples: int = 1) -> HealthData:
        return HealthData(weather_data.time_period, np.random.normal(self._case_average, self._case_std, n_samples))
