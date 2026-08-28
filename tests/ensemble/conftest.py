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


class RecordingPredictor(ConstantPredictor):
    """Constant predictor that records the columns present in each future_data it sees."""

    def __init__(self, value: float, n_samples: int, seen_future_fields: list[list[str]]):
        super().__init__(value, n_samples)
        self.seen_future_fields = seen_future_fields

    def predict(self, historic_data, future_data):
        self.seen_future_fields.append(list(future_data.to_pandas().columns))
        return super().predict(historic_data, future_data)


class RecordingEstimator:
    def __init__(self, value: float, n_samples: int, seen_future_fields: list[list[str]]):
        self._value = value
        self._n_samples = n_samples
        self._seen_future_fields = seen_future_fields

    def train(self, _train_data):
        return RecordingPredictor(self._value, self._n_samples, self._seen_future_fields)


class RecordingTemplate(DummyTemplate):
    """Template whose predictors share one record of the future_data columns seen."""

    def __init__(self, value: float, n_samples: int, name: str):
        self.seen_future_fields: list[list[str]] = []
        super().__init__(
            lambda: RecordingEstimator(value, n_samples, self.seen_future_fields),
            name,
        )


@pytest.fixture
def recording_template_factory():
    def _make(value: float, n_samples: int, name: str):
        return RecordingTemplate(value, n_samples, name)

    return _make


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


class PointForecastOnly:
    """Predictions carrying only a point forecast column, without the merge keys.

    A predictor shaped like this is what the row-order fallback in reshape_samples
    exists for; ``Samples`` itself always carries a time_period column.
    """

    def __init__(self, values, column: str = "forecast"):
        self._values = list(values)
        self._column = column

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame({self._column: self._values})


@pytest.fixture
def point_forecast_only_factory():
    def _make(values, column: str = "forecast"):
        return PointForecastOnly(values, column)

    return _make
