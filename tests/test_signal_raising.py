"""Forecast signal selection and persistence, using real stored history and strategies."""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from sqlmodel import select

from chap_core.database.signal_tables import SignalReleaseStatus
from chap_core.database.database import SessionWrapper
from chap_core.database.tables import Prediction
from chap_core.datatypes import Samples
from chap_core.rest_api.db_worker_functions import run_prediction
from chap_core.services.signal_service import list_signals, raise_signals_for_prediction
from chap_core.services.threshold_service import InvalidThresholdInputError
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange


def test_without_policy_creates_no_signals(signal_prediction):
    session, prediction = signal_prediction
    prediction.threshold_policy_id = None
    session.commit()
    assert raise_signals_for_prediction(session, prediction.id) == []
    assert list_signals(session) == []


def test_most_severe_firing_level_per_cell(signal_prediction):
    session, prediction = signal_prediction
    signals = raise_signals_for_prediction(session, prediction.id)
    assert {(a.org_unit, a.time_period, a.level) for a in signals} == {
        ("A", "2024-01", "level_1"),
        ("A", "2024-02", "level_2"),
    }
    assert all(a.release_status is SignalReleaseStatus.PENDING for a in signals)


def test_severity_uses_policy_order(signal_prediction):
    session, prediction = signal_prediction
    policy = prediction.prediction_setup.threshold_policy
    policy.levels = list(reversed(policy.levels))
    session.commit()
    signals = raise_signals_for_prediction(session, prediction.id)
    assert [a.level for a in signals] == ["level_1", "level_1"]


def test_thresholds_use_backtest_target_column(signal_prediction):
    session, prediction = signal_prediction
    prediction.prediction_setup.backtest.specification.target_column = "hospitalisations"
    for observation in prediction.dataset.observations:
        observation.feature_name = "hospitalisations"
    session.commit()
    signals = raise_signals_for_prediction(session, prediction.id)
    assert {(a.org_unit, a.time_period, a.level) for a in signals} == {
        ("A", "2024-01", "level_1"),
        ("A", "2024-02", "level_2"),
    }


@pytest.mark.parametrize("cutoff, fires", [(0.5, True), (0.5001, False)])
def test_exceedance_cutoff_is_inclusive_but_samples_must_be_above_line(signal_prediction, cutoff, fires):
    session, prediction = signal_prediction
    policy = prediction.prediction_setup.threshold_policy
    policy.levels = [policy.levels[0].model_copy(update={"alarm_threshold": cutoff})]
    session.commit()
    signals = raise_signals_for_prediction(session, prediction.id)
    # Exactly half of January A's samples are strictly above the media threshold of 15.
    assert any(a.org_unit == "A" and a.time_period == "2024-01" for a in signals) is fires


def test_missing_season_history_is_skipped_with_warning(signal_prediction, caplog):
    session, prediction = signal_prediction
    for observation in list(prediction.dataset.observations):
        if observation.period.endswith("-01"):
            session.delete(observation)
    session.commit()
    with caplog.at_level(logging.WARNING):
        signals = raise_signals_for_prediction(session, prediction.id)
    assert [(a.org_unit, a.time_period) for a in signals] == [("A", "2024-02")]
    assert "Skipping threshold level" in caplog.text


def test_level_without_thresholds_fails_signal_raising(signal_prediction):
    session, prediction = signal_prediction
    with patch(
        "chap_core.assessment.thresholds.percentile.PercentileThresholdStrategy.compute",
        side_effect=ValueError("failed"),
    ):
        with pytest.raises(InvalidThresholdInputError):
            raise_signals_for_prediction(session, prediction.id)
    assert list_signals(session) == []


def _run_prediction(session, original, before_forecast=lambda: None):
    forecast = DataSet({"A": Samples(PeriodRange.from_strings(["2024-01"]), np.array([[21.0, 21.0]]))})

    def forecast_ahead(*args, **kwargs):
        before_forecast()
        return forecast

    wrapper = SessionWrapper(session=session)
    with (
        patch.object(wrapper, "get_configured_model_with_code"),
        patch("chap_core.rest_api.db_worker_functions.forecast_ahead", forecast_ahead),
    ):
        return run_prediction(
            original.model_id, str(original.dataset_id), 1, "new", wrapper, original.prediction_setup_id
        )


def test_run_prediction_keeps_the_policy_captured_at_start(signal_prediction):
    session, original = signal_prediction
    policy_id = original.threshold_policy_id

    def clear_setup_policy():
        original.prediction_setup.threshold_policy = None
        session.commit()

    prediction_id = _run_prediction(session, original, before_forecast=clear_setup_policy)
    assert session.get(Prediction, prediction_id).threshold_policy_id == policy_id
    assert [a.level for a in list_signals(session, prediction_id=prediction_id)] == ["level_2"]


def test_run_prediction_fails_but_keeps_prediction_when_signals_fail(signal_prediction):
    session, original = signal_prediction
    with patch("chap_core.rest_api.db_worker_functions.raise_signals_for_prediction", side_effect=RuntimeError):
        with pytest.raises(RuntimeError):
            _run_prediction(session, original)
    stored = session.exec(select(Prediction).where(Prediction.name == "new")).one()
    assert stored.forecasts[0].values == [21.0, 21.0]


def test_run_prediction_stores_every_baseline_forecast(signal_prediction):
    session, original = signal_prediction
    prediction_id = _run_prediction(session, original)
    prediction = session.get(Prediction, prediction_id)
    baselines = prediction.baseline_predictions
    assert {baseline.model_id for baseline in baselines} == {"global_median", "persistence", "seasonal_median"}
    assert {baseline.prediction_setup_id for baseline in baselines} == {None}
    assert baselines[0].forecasts[0].values == [21.0, 21.0]


def test_run_prediction_keeps_prediction_and_other_baselines_when_a_baseline_fails(signal_prediction):
    session, original = signal_prediction
    forecast = DataSet({"A": Samples(PeriodRange.from_strings(["2024-01"]), np.array([[21.0, 21.0]]))})
    wrapper = SessionWrapper(session=session)
    with (
        patch.object(wrapper, "get_configured_model_with_code"),
        patch(
            "chap_core.rest_api.db_worker_functions.forecast_ahead",
            side_effect=[forecast, forecast, RuntimeError("baseline bug"), forecast],
        ),
    ):
        prediction_id = run_prediction(
            original.model_id, str(original.dataset_id), 1, "new", wrapper, original.prediction_setup_id
        )
    assert len(session.get(Prediction, prediction_id).baseline_predictions) == 2
