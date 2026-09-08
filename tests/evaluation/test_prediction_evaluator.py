from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from chap_core.assessment.prediction_evaluator import IncompleteBacktestError, backtest
from chap_core.datatypes import Samples
from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def test_backtest_passes_stride_to_train_test_generator():
    mock_estimator = MagicMock()
    mock_data = MagicMock()

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = (MagicMock(), iter(_splits(4)))

        list(backtest(mock_estimator, mock_data, prediction_length=3, n_test_sets=4, stride=2))

        mock_ttg.assert_called_once_with(mock_data, 3, 4, stride=2, future_weather_provider=None)


def _splits(n):
    """Build ``n`` (historic, future, truth) split tuples with distinguishable historic data."""
    return [(f"historic{i}", f"future{i}", MagicMock()) for i in range(n)]


def test_backtest_trains_once_by_default():
    mock_estimator = MagicMock()

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = ("train_set", iter(_splits(4)))

        list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=4, stride=1))

    assert mock_estimator.train.call_count == 1
    # Split 0 trains on the dedicated train_set, preserving the single-train behaviour.
    mock_estimator.train.assert_called_once_with("train_set")


def test_backtest_retrains_at_evenly_spaced_splits():
    mock_estimator = MagicMock()

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = ("train_set", iter(_splits(4)))

        list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=4, stride=1, n_retrain=2))

    assert mock_estimator.train.call_count == 2
    trained_on = [call.args[0] for call in mock_estimator.train.call_args_list]
    # Split 0 uses train_set; the halfway retrain (split 2) uses its expanding historic window.
    assert trained_on == ["train_set", "historic2"]


N_SPLITS = 3


def _drop_location(forecasts, location):
    return DataSet({loc: samples for loc, samples in forecasts.items() if loc != location})


def _nan_samples(forecasts, location):
    samples = forecasts[location]
    tampered = Samples(samples.time_period, np.full(np.shape(samples.samples), np.nan))
    return DataSet({loc: tampered if loc == location else s for loc, s in forecasts.items()})


def _empty_samples(forecasts, location):
    samples = forecasts[location]
    tampered = Samples(samples.time_period, np.zeros((len(samples.time_period), 0)))
    return DataSet({loc: tampered if loc == location else s for loc, s in forecasts.items()})


def _no_forecasts(forecasts, location):
    return None


class _TamperingEstimator:
    """Naive estimator whose predictor rewrites the forecast for one location on one split."""

    def __init__(self, tamper, location, split_index):
        self.tamper = tamper
        self.location = location
        self.split_index = split_index
        self.predict_calls = 0

    def train(self, data):
        inner = NaiveEstimator().train(data)
        estimator = self

        class _Predictor:
            def predict(self, historic_data, future_data):
                forecasts = inner.predict(historic_data, future_data)
                split_index = estimator.predict_calls
                estimator.predict_calls += 1
                if split_index != estimator.split_index:
                    return forecasts
                return estimator.tamper(forecasts, estimator.location)

        return _Predictor()


def _run_backtest(estimator, dataset):
    return list(backtest(estimator, dataset, prediction_length=2, n_test_sets=N_SPLITS))


def test_backtest_passes_when_every_org_unit_is_forecast_in_every_split(health_population_data):
    results = _run_backtest(NaiveEstimator(), health_population_data)

    assert len(results) == N_SPLITS
    for split_result in results:
        assert set(split_result.locations()) == set(health_population_data.locations())


def test_backtest_raises_when_org_unit_is_dropped_in_a_single_split(health_population_data):
    # Dropping a location in only one split must fail: a union over all splits would not catch it.
    locations = list(health_population_data.locations())
    estimator = _TamperingEstimator(_drop_location, locations[0], split_index=1)

    with pytest.raises(IncompleteBacktestError) as excinfo:
        _run_backtest(estimator, health_population_data)

    message = str(excinfo.value)
    assert locations[0] in message
    assert f"split 2 of {N_SPLITS}" in message
    assert f"missing 1 of {len(locations)} org units" in message


@pytest.mark.parametrize("tamper", [_nan_samples, _empty_samples])
def test_backtest_raises_when_org_unit_has_unusable_samples(health_population_data, tamper):
    location = next(iter(health_population_data.locations()))
    estimator = _TamperingEstimator(tamper, location, split_index=0)

    with pytest.raises(IncompleteBacktestError, match="empty or non-finite samples") as excinfo:
        _run_backtest(estimator, health_population_data)

    assert location in str(excinfo.value)


def test_backtest_raises_when_predictor_returns_none_for_a_split(health_population_data):
    estimator = _TamperingEstimator(_no_forecasts, location=None, split_index=2)

    with pytest.raises(IncompleteBacktestError, match=f"returned no forecasts for split 3 of {N_SPLITS}"):
        _run_backtest(estimator, health_population_data)
