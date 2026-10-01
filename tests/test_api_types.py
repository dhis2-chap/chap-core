import pytest
from pydantic import ValidationError

from chap_core.api_types import BacktestParams


def test_backtest_params_n_retrain_defaults_to_one():
    params = BacktestParams(n_periods=3, n_splits=7, stride=1)
    assert params.n_retrain == 1


def test_backtest_params_rejects_n_retrain_above_n_splits():
    with pytest.raises(ValidationError):
        BacktestParams(n_periods=3, n_splits=4, stride=1, n_retrain=5)


@pytest.mark.parametrize("name", ["n_periods", "n_splits", "stride", "n_retrain"])
def test_backtest_params_accepts_minimum(name):
    assert getattr(BacktestParams(**{name: 1}), name) == 1


@pytest.mark.parametrize("name", ["n_periods", "n_splits", "stride", "n_retrain"])
@pytest.mark.parametrize("value", [0, -1, 0.5])
def test_backtest_params_rejects_invalid_counts(name, value):
    with pytest.raises(ValidationError):
        BacktestParams(**{name: value})
