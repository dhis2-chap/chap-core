import dataclasses
from collections import defaultdict

import numpy as np
from sklearn import linear_model

from .datatypes import ClimateData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import PeriodRange



class MonthlyClimatePredictor:
    def __init__(self):
        self._models = defaultdict(dict)
        self._cls = None

    def _feature_matrix(self, time_period: PeriodRange):
        return time_period.month[:,None] == np.arange(1, 13)

    def train(self, train_data: SpatioTemporalDict[ClimateData]):
        for location, data in train_data.items():
            data = data.data()
            self._cls = data.__class__
            x = self._feature_matrix(data.time_period)
            for field in dataclasses.fields(data):
                if field.name == 'time_period':
                    continue
                y = getattr(data, field.name)
                model = linear_model.LinearRegression()
                model.fit(x, y[:,None])
                self._models[location][field.name] = model

    def predict(self, time_period: PeriodRange):
        x = self._feature_matrix(time_period)
        prediction_dict = {}
        for location, models in self._models.items():
            prediction_dict[location] = self._cls(time_period, **{field: model.predict(x) for field, model in models.items()})
        return SpatioTemporalDict(prediction_dict)





