"""
This is simulation code that can useful for creating tests. Currently in use.

todo: Can maybe be moved to tests/
"""

import abc
from typing import Any

import pydantic
import numpy as np
from numpy.random import normal, poisson

from chap_core.database.dataset_tables import DataSet, Observation
from chap_core.database.tables import BackTest, BackTestForecast


class SimulationParams(pydantic.BaseModel):
    loc: float = 5
    scale: float = 2


class DatasetDimensions(pydantic.BaseModel):
    locations: list[str]
    time_periods: list[str]
    target: str = "disease_cases"
    features: list[str] = []


class Simulator(abc.ABC):
    def __init__(self, params: SimulationParams):
        self._params = params

    def simulate(self, data_dims: DatasetDimensions) -> DataSet: ...


class AdditiveSimulator(Simulator):
    def __init__(self, params: SimulationParams = SimulationParams()):
        self._params = params

    def generate_raw(self, data_dims: DatasetDimensions) -> np.ndarray:
        location_offsets = normal(0, 1, len(data_dims.locations))
        x = np.arange(len(data_dims.time_periods)) / 12 * 2 * np.pi
        time_pattern = np.sin(x)
        mu = location_offsets[:, None] + time_pattern[None, :]
        with_noise = normal(mu, 0.1)
        values = with_noise * self._params.scale + self._params.loc
        return values

    def simulate(self, data_dims: DatasetDimensions) -> DataSet:
        feature_names = data_dims.features + [data_dims.target]
        observations = []
        for feature_name in feature_names:
            observations.extend(self.simulate_observations(data_dims, feature_name))
        return DataSet(
            name="Simulated DataSet", covariates=data_dims.features + [data_dims.target], observations=observations
        )

    def simulate_observations(self, data_dims: DatasetDimensions, feature_name) -> list[Observation]:
        values = self.generate_raw(data_dims)
        values = np.exp(values).astype(int)
        observations = [
            Observation(
                period=data_dims.time_periods[time_idx],
                org_unit=data_dims.locations[loc_idx],
                feature_name=feature_name,
                value=int(values[loc_idx, time_idx]),
            )
            for loc_idx in range(len(data_dims.locations))
            for time_idx in range(len(data_dims.time_periods))
        ]
        return observations


class ForecastParams(pydantic.BaseModel):
    prediction_length: int = 3
    n_samples: int = 100
    n_splits: int = 2


class BacktestSimulator:
    def __init__(self, params: ForecastParams = ForecastParams()):
        self._params = params

    def simulate(self, dataset: DataSet, dataset_dims: DatasetDimensions) -> BackTest:
        periods = dataset_dims.time_periods[-(self._params.prediction_length + self._params.n_splits - 1) :]
        split_periods = periods[: self._params.n_splits]
        backtest = BackTest(
            dataset=dataset, model_id="Naive Forecast", org_units=dataset_dims.locations, split_periods=split_periods
        )
        forecasts = []
        for i in range(self._params.n_splits):
            forecasts.extend(self.simulate_split(dataset, periods[i : i + self._params.prediction_length]))
        backtest.forecasts = forecasts
        return backtest

    def simulate_split(self, dataset: DataSet, periods: list[str]) -> list[Any]:
        forecasts = []
        split_period = periods[0]
        for observation in dataset.observations:
            if observation.period not in periods:
                continue

            rate = normal(observation.value, observation.value, size=self._params.n_samples)
            rate = np.maximum(rate, 0)
            samples = poisson(rate).astype(float)
            forecasts.append(
                BackTestForecast(
                    values=samples.tolist(),
                    last_seen_period=split_period,
                    last_train_period=split_period,
                    **observation.model_dump(),
                )
            )
        return forecasts
