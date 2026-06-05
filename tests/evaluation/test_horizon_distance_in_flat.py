from __future__ import annotations

import pandas as pd

from chap_core.assessment.evaluation import Evaluation


def test_to_flat_populated_backtest_includes_horizon_distance(backtest):
    """Ensure Evaluation.to_flat always exposes `horizon_distance` as int for normal backtests."""
    evaluation = Evaluation.from_backtest(backtest)
    flat = evaluation.to_flat()

    forecasts_df = pd.DataFrame(flat.forecasts)
    assert "horizon_distance" in forecasts_df.columns
    assert pd.api.types.is_integer_dtype(forecasts_df["horizon_distance"]) or (
        forecasts_df["horizon_distance"].dtype == object
    )
    # If there are rows, distances should be >= 1
    if not forecasts_df.empty:
        assert int(forecasts_df["horizon_distance"].min()) >= 1


def test_to_flat_empty_backtest_has_horizon_distance_column(backtest_empty):
    """Empty backtests should still expose the `horizon_distance` column (possibly empty)."""
    evaluation = Evaluation.from_backtest(backtest_empty)
    flat = evaluation.to_flat()

    forecasts_df = pd.DataFrame(flat.forecasts)
    # Column must exist even if there are no rows
    assert "horizon_distance" in forecasts_df.columns
