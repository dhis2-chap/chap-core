import dataclasses
from collections import defaultdict

import numpy as np
from sklearn import linear_model

from .datatypes import ClimateData, SimpleClimateData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange, Month, Week


def get_climate_predictor(train_data: DataSet[ClimateData]):
    if isinstance(train_data.period_range[0], Month):
        estimator = MonthlyClimatePredictor()
    else:
        assert isinstance(train_data.period_range[0], Week)
        estimator = WeeklyClimatePredictor()
    estimator.train(train_data)
    return estimator


class MonthlyClimatePredictor:
    def __init__(self):
        self._models = defaultdict(dict)
        self._cls = None

    def _feature_matrix(self, time_period: PeriodRange):
        return time_period.month[:, None] == np.arange(1, 13)

    def train(self, train_data: DataSet[ClimateData]):
        train_data = train_data.remove_field("disease_cases")
        for location, data in train_data.items():
            self._cls = data.__class__
            x = self._feature_matrix(data.time_period)
            for field in dataclasses.fields(data):
                if field.name in ("time_period"):
                    continue
                y = getattr(data, field.name)
                model = linear_model.LinearRegression()
                model.fit(x, y[:, None])
                self._models[location][field.name] = model

    def predict(self, time_period: PeriodRange):
        x = self._feature_matrix(time_period)
        prediction_dict = {}
        for location, models in self._models.items():
            prediction_dict[location] = self._cls(
                time_period,
                **{field: model.predict(x).ravel() for field, model in models.items()},
            )
        return DataSet(prediction_dict)


class WeeklyClimatePredictor(MonthlyClimatePredictor):
    def _feature_matrix(self, time_period: PeriodRange):
        t = time_period.week[:, None] == np.arange(1, 53)
        t[..., -1] |= time_period.week == 53
        return t


class FutureWeatherFetcher:
    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]: ...


class SeasonalForecastFetcher:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]: ...


class QuickForecastFetcher:
    def __init__(self, historical_data: DataSet[SimpleClimateData]):
        self._climate_predictor = get_climate_predictor(historical_data)

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        return self._climate_predictor.predict(period_range)
    
class FetcherNd:
    def __init__(self, historical_data: DataSet[SimpleClimateData]):
        self.historical_data = historical_data
        self._cls = list(historical_data.values())[0].__class__

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        prediction_dict = {}
        for location, data in self.historical_data.items():
            prediction_dict[location] = self._cls(
                period_range,
                **{field.name: getattr(data, field.name)[-len(period_range):] for field in dataclasses.fields(data) if field.name not in ("time_period" ,)},
            )

        return DataSet(prediction_dict)
