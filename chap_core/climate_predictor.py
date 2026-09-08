import dataclasses
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn import linear_model

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month, PeriodRange, Week

from .datatypes import ClimateData, SimpleClimateData


def get_climate_predictor(train_data: DataSet[ClimateData]):
    first = train_data.period_range[0]
    if isinstance(first, Month):
        estimator = MonthlyClimatePredictor()
    elif isinstance(first, Week):
        estimator = WeeklyClimatePredictor()
    else:
        raise ValueError(
            f"The climatology fit needs monthly or weekly data, got {type(first).__name__}. "
            "Use the 'observed' future-weather provider for other resolutions."
        )
    estimator.train(train_data)
    return estimator


class MonthlyClimatePredictor:
    def __init__(self):
        self._models: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        self._cls: type | None = None

    def _feature_matrix(self, time_period: PeriodRange):
        return time_period.month[:, None] == np.arange(1, 13)

    def train(self, train_data: DataSet[ClimateData]):
        train_data = train_data.remove_field("disease_cases")
        for location, data in train_data.items():
            self._cls = data.__class__
            # Retain the forecast grid even when there are no covariates to fit.
            self._models[location] = {}
            x = self._feature_matrix(data.time_period)
            for field in dataclasses.fields(data):  # type: ignore[arg-type]
                if field.name in ("time_period"):
                    continue
                y = np.asarray(getattr(data, field.name), dtype=float)
                # Fit on the observed rows only; a gap in a covariate should not
                # take the whole evaluation down.
                observed = ~np.isnan(y)
                if not observed.any():
                    raise ValueError(
                        f"Cannot fit a climatology for '{field.name}' in {location}: every value is missing."
                    )
                model = linear_model.LinearRegression()
                model.fit(x[observed], y[observed, None])
                self._models[location][field.name] = model

    def predict(self, time_period: PeriodRange):
        x = self._feature_matrix(time_period)
        prediction_dict = {}
        assert self._cls is not None, "Model not trained - call train() first"
        for location, models in self._models.items():
            fields = {field: model.predict(x).ravel() for field, model in models.items()}
            prediction_dict[location] = self._cls(time_period, **fields)
        return DataSet(prediction_dict)


class WeeklyClimatePredictor(MonthlyClimatePredictor):
    def _feature_matrix(self, time_period: PeriodRange):
        t = time_period.week[:, None] == np.arange(1, 53)
        t[..., -1] |= time_period.week == 53
        return t


class FutureWeatherFetcher:
    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        raise NotImplementedError


class SeasonalForecastFetcher:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        raise NotImplementedError


class QuickForecastFetcher:
    def __init__(self, historical_data: DataSet[SimpleClimateData]):
        self._climate_predictor = get_climate_predictor(historical_data)  # type: ignore[arg-type]

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        return self._climate_predictor.predict(period_range)  # type: ignore[no-any-return]


class FetcherNd:
    def __init__(self, historical_data: DataSet[SimpleClimateData]):
        self.historical_data = historical_data
        self._cls = next(iter(historical_data.values())).__class__

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        prediction_dict = {}
        for location, data in self.historical_data.items():
            prediction_dict[location] = self._cls(  # type: ignore[call-arg]
                period_range,
                **{
                    field.name: getattr(data, field.name)[-len(period_range) :]
                    for field in dataclasses.fields(data)  # type: ignore[arg-type]
                    if field.name != "time_period"
                },
            )

        return DataSet(prediction_dict)
