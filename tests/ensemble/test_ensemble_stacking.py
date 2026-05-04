import numpy as np

from chap_core.datatypes import Samples
from chap_core.ensemble.ensemble_model import EnsembleModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class _DummyTemplate:
    def __init__(self, estimator_cls, name: str):
        self._estimator_cls = estimator_cls
        self.name = name

    def get_model(self, _config):
        return self._estimator_cls


class _ConstantPredictor:
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


class _ConstantEstimator:
    def __init__(self, value: float, n_samples: int):
        self._value = value
        self._n_samples = n_samples

    def train(self, _train_data):
        return _ConstantPredictor(self._value, self._n_samples)


def test_deterministic_predict_shape_and_weights(weekly_full_data):
    templates = [
        _DummyTemplate(lambda: _ConstantEstimator(5.0, 1), "model_a"),
        _DummyTemplate(lambda: _ConstantEstimator(10.0, 1), "model_b"),
    ]
    model = EnsembleModel(base_templates=templates, method="deterministic", n_samples=5)

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    assert model.weights is not None
    assert len(model.weights) == 2
    assert np.isclose(float(np.sum(model.weights)), 100.0, atol=1e-6)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == 1
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)


def test_deterministic_residual_bootstrap_generates_samples(weekly_full_data):
    templates = [
        _DummyTemplate(lambda: _ConstantEstimator(2.0, 1), "model_a"),
        _DummyTemplate(lambda: _ConstantEstimator(4.0, 1), "model_b"),
    ]
    n_samples = 4
    model = EnsembleModel(
        base_templates=templates,
        method="deterministic",
        n_samples=n_samples,
        use_residual_bootstrap=True,
        random_state=123,
    )

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == n_samples
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)


def test_probabilistic_predict_samples_count(weekly_full_data):
    templates = [
        _DummyTemplate(lambda: _ConstantEstimator(3.0, 2), "model_a"),
        _DummyTemplate(lambda: _ConstantEstimator(6.0, 2), "model_b"),
    ]
    n_samples = 6
    model = EnsembleModel(base_templates=templates, method="probabilistic", n_samples=n_samples, random_state=7)

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == n_samples
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)
