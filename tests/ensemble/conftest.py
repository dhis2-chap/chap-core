import numpy as np
import pytest

from chap_core.datatypes import Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class DummyTemplate:
    def __init__(self, estimator_cls, name: str):
        self._estimator_cls = estimator_cls
        self.name = name

    def get_model(self, _config):
        return self._estimator_cls


class ConstantPredictor:
    def __init__(self, value: float, n_samples: int):
        self._value = value
        self._n_samples = n_samples

    def predict(self, _historic_data, future_data):
        result = {}
        for loc in future_data.locations():
            tp = future_data[loc].time_period
            vals = np.full(len(tp), self._value, dtype=float)
            samples = np.tile(vals.reshape(-1, 1), (1, self._n_samples))
            result[loc] = Samples(tp, samples)
        return DataSet(result)


class ConstantEstimator:
    def __init__(self, value: float, n_samples: int):
        self._value = value
        self._n_samples = n_samples

    def train(self, _train_data):
        return ConstantPredictor(self._value, self._n_samples)


class NaNPredictor(ConstantPredictor):
    def __init__(self, value: float, n_samples: int, nan_index: int = 0):
        super().__init__(value, n_samples)
        self._nan_index = nan_index

    def predict(self, _historic_data, future_data):
        result = {}
        for loc in future_data.locations():
            tp = future_data[loc].time_period
            vals = np.full(len(tp), self._value, dtype=float)
            if len(vals) > 0:
                vals[self._nan_index] = np.nan
            samples = np.tile(vals.reshape(-1, 1), (1, self._n_samples))
            result[loc] = Samples(tp, samples)
        return DataSet(result)


class NaNEstimator:
    def __init__(self, value: float, n_samples: int, nan_index: int = 0):
        self._value = value
        self._n_samples = n_samples
        self._nan_index = nan_index

    def train(self, _train_data):
        return NaNPredictor(self._value, self._n_samples, self._nan_index)


@pytest.fixture
def constant_template_factory():
    def _make(value: float, n_samples: int, name: str):
        return DummyTemplate(lambda: ConstantEstimator(value, n_samples), name)

    return _make


@pytest.fixture
def nan_template_factory():
    def _make(value: float, n_samples: int, name: str, nan_index: int = 0):
        return DummyTemplate(lambda: NaNEstimator(value, n_samples, nan_index), name)

    return _make


@pytest.fixture
def constant_predictor_factory():
    def _make(value: float, n_samples: int):
        return ConstantPredictor(value, n_samples)

    return _make


@pytest.fixture
def base_residuals_factory(weekly_full_data):
    def _make(value: float):
        location = next(iter(weekly_full_data.locations()))
        series = weekly_full_data[location]
        return np.asarray(series.disease_cases, float) - value

    return _make


@pytest.fixture
def vincentization_samples(weekly_full_data):
    location = next(iter(weekly_full_data.locations()))
    series = weekly_full_data[location]
    base = np.asarray(series.disease_cases, float)
    base = base[np.isfinite(base)]

    n_samples = 5
    x1 = np.tile(base.reshape(-1, 1), (1, n_samples))
    x2 = np.tile((base + 2.0).reshape(-1, 1), (1, n_samples))

    perm = np.array([2, 4, 1, 0, 3])
    x1_perm = x1[:, perm]
    x2_perm = x2[:, perm[::-1]]
    weights = np.array([0.3, 0.7], dtype=float)
    return x1, x2, x1_perm, x2_perm, weights
