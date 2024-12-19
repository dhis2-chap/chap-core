import dataclasses

import numpy as np
from sklearn import linear_model

from chap_core.spatio_temporal_data.temporal_dataclass import (
    DataSet,
    TemporalDataclass,
)
from chap_core.datatypes import HealthData, ClimateHealthTimeSeries, ClimateData


class NaivePredictor:
    """This should be a linear regression of prev cases and season"""

    def __init__(self, lead_time=1):
        self._average_cases = None

    def train(self, data: ClimateHealthTimeSeries):
        self._average_cases = data.disease_cases.mean()

    def predict(self, future_climate_data: ClimateData) -> HealthData:
        return HealthData(
            future_climate_data.time_period,
            np.full(len(future_climate_data), self._average_cases),
        )


class MultiRegionNaivePredictor:
    """TODO: This should be a linear regression of prev cases and season for each location."""

    def __init__(self, *args, **kwargs):
        self._training_stop = None
        self._average_cases = None

    def _get_mean(self, data):
        y = data.data().disease_cases
        y = y[~np.isnan(y)]
        return y.mean()
        # return data.data().disease_cases.mean()

    def train(self, data: DataSet[ClimateHealthTimeSeries]):
        self._average_cases = {location: self._get_mean(data) for location, data in data.items()}
        # self._buffer = next(iter(data.values())).time_period[-1]

    def predict(self, future_weather: DataSet[ClimateData]) -> HealthData:
        prediction_dict = {
            location: HealthData(entry.data().time_period[:1], np.full(1, self._average_cases[location]))
            for location, entry in future_weather.items()
        }
        return DataSet(prediction_dict)


class MultiRegionPoissonModel:
    def __init__(self, *args, **kwargs):
        self._training_stop = None
        self._models = {}
        self._saved_state = {}

    def _create_feature_matrix(self, data: ClimateHealthTimeSeries):
        data = data.data()
        lagged_values = data.disease_cases[:-1, None]
        month = np.array([period.month for period in data.time_period])
        season = month[1:, None] == np.arange(1, 13)
        return np.hstack([lagged_values, season])

    def train(self, data: DataSet[ClimateHealthTimeSeries]):
        for location, location_data in data.items():
            X = self._create_feature_matrix(location_data)
            y = location_data.data().disease_cases[1:]
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            assert mask[-1]
            X = X[mask]
            y = y[mask]
            model = linear_model.PoissonRegressor()
            model.fit(X, y)
            self._models[location] = model

            saved_data = location_data.data()[-1:]
            assert not np.any(np.isnan(saved_data.disease_cases)), f"{saved_data.disease_cases}"
            self._saved_state[location] = TemporalDataclass(saved_data)

    def predict(self, data: DataSet[ClimateData]) -> DataSet[HealthData]:
        prediction_dict = {}
        for location, location_data in data.items():
            state_values = self._saved_state[location]
            # state_values = TemporalDataclass(location_data.data().__class__(**{field.name: getattr(state_values.data(), field.name) for field in dataclasses.fields(location_data.data())}))
            location_data = TemporalDataclass(
                state_values.data().__class__(
                    **{
                        field.name: getattr(location_data.data(), field.name)
                        for field in dataclasses.fields(location_data.data())
                    }
                    | {"disease_cases": np.full(len(location_data.data()), 0)}
                )
            )
            # location_data.data().disease_cases = np.full(len(location_data.data()), np.nan)
            X = self._create_feature_matrix(state_values.join(location_data))
            prediction = self._models[location].predict(X[-1:])
            prediction_dict[location] = HealthData(location_data.data().time_period[:1], np.atleast_1d(prediction))

        return DataSet(prediction_dict)


class NaiveForecastSampler:
    def __init__(self):
        self._case_average = None
        self._case_std = None

    def train(self, time_series: ClimateHealthTimeSeries):
        self._case_average = time_series.disease_cases.mean()
        self._case_std = time_series.disease_cases.std()

    def sample(self, weather_data: ClimateData, n_samples: int = 1) -> HealthData:
        return HealthData(
            weather_data.time_period,
            np.random.normal(self._case_average, self._case_std, n_samples),
        )
