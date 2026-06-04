import pytest
from pydantic import ValidationError

from chap_core.api_types import BacktestParams


def test_backtest_params_n_retrain_defaults_to_one():
    params = BacktestParams(n_periods=3, n_splits=7, stride=1)
    assert params.n_retrain == 1


def test_backtest_params_rejects_n_retrain_above_n_splits():
    with pytest.raises(ValidationError):
        BacktestParams(n_periods=3, n_splits=4, stride=1, n_retrain=5)
