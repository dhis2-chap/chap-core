"""Forecast alert selection and persistence, using real stored history and strategies."""

import logging
from unittest.mock import patch

import pytest
from sqlmodel import select

from chap_core.database.alert_tables import AlertApproval
from chap_core.database.database import SessionWrapper
from chap_core.database.tables import Prediction, PredictionInfo
from chap_core.rest_api.db_worker_functions import run_prediction
from chap_core.services.alert_service import list_alerts, raise_alerts_for_prediction
from chap_core.services.threshold_service import InvalidThresholdInputError


def test_without_policy_creates_no_alerts(alert_prediction):
    session, prediction = alert_prediction
    prediction.alert_policy_id = None
    session.commit()
    assert raise_alerts_for_prediction(session, prediction.id) == []
    assert list_alerts(session) == []


def test_most_severe_firing_level_per_cell(alert_prediction):
    session, prediction = alert_prediction
    policy_id = prediction.alert_policy_id
    prediction.prediction_setup.alert_policy = None
    session.commit()
    alerts = raise_alerts_for_prediction(session, prediction.id)
    assert {(a.org_unit, a.time_period, a.level) for a in alerts} == {
        ("A", "2024-01", "monitor"),
        ("A", "2024-02", "action"),
    }
    assert all(a.approved is AlertApproval.PENDING for a in alerts)
    assert all(a.alert_policy_id == policy_id for a in alerts)


def test_severity_uses_policy_order(alert_prediction):
    session, prediction = alert_prediction
    policy = prediction.prediction_setup.alert_policy
    policy.levels = list(reversed(policy.levels))
    session.commit()
    alerts = raise_alerts_for_prediction(session, prediction.id)
    assert [a.level for a in alerts] == ["monitor", "monitor"]


def test_thresholds_use_backtest_target_column(alert_prediction):
    session, prediction = alert_prediction
    prediction.prediction_setup.backtest.specification.target_column = "hospitalisations"
    for observation in prediction.dataset.observations:
        observation.feature_name = "hospitalisations"
    session.commit()
    alerts = raise_alerts_for_prediction(session, prediction.id)
    assert {(a.org_unit, a.time_period, a.level) for a in alerts} == {
        ("A", "2024-01", "monitor"),
        ("A", "2024-02", "action"),
    }


@pytest.mark.parametrize("cutoff, fires", [(0.5, True), (0.5001, False)])
def test_exceedance_cutoff_is_inclusive_but_samples_must_be_above_line(alert_prediction, cutoff, fires):
    session, prediction = alert_prediction
    policy = prediction.prediction_setup.alert_policy
    policy.levels = [policy.levels[0].model_copy(update={"exceedance_threshold": cutoff})]
    session.commit()
    alerts = raise_alerts_for_prediction(session, prediction.id)
    # Exactly half of January A's samples are strictly above the median threshold of 15.
    assert any(a.org_unit == "A" and a.time_period == "2024-01" for a in alerts) is fires


def test_missing_season_history_is_skipped_with_warning(alert_prediction, caplog):
    session, prediction = alert_prediction
    for observation in list(prediction.dataset.observations):
        if observation.period.endswith("-01"):
            session.delete(observation)
    session.commit()
    with caplog.at_level(logging.WARNING):
        alerts = raise_alerts_for_prediction(session, prediction.id)
    assert [(a.org_unit, a.time_period) for a in alerts] == [("A", "2024-02")]
    assert "Skipping alert level" in caplog.text


def test_level_without_thresholds_fails_alert_raising(alert_prediction):
    session, prediction = alert_prediction
    with patch(
        "chap_core.assessment.thresholds.percentile.PercentileThresholdStrategy.compute",
        side_effect=ValueError("failed"),
    ):
        with pytest.raises(InvalidThresholdInputError):
            raise_alerts_for_prediction(session, prediction.id)
    assert list_alerts(session) == []


def run_stored_prediction(session, original, forecast, during_forecast=None):
    wrapper = SessionWrapper(session=session)
    with (
        patch.object(wrapper, "get_configured_model_with_code"),
        patch(
            "chap_core.rest_api.db_worker_functions.forecast_ahead", return_value=forecast, side_effect=during_forecast
        ),
    ):
        return run_prediction(
            original.model_id, str(original.dataset_id), 1, "new", wrapper, original.prediction_setup_id
        )


@pytest.mark.parametrize("has_policy, fires", [(True, True), (True, False), (False, True)])
def test_run_prediction_captures_policy_before_forecasting(alert_prediction, alert_forecast, has_policy, fires):
    session, original = alert_prediction
    setup = original.prediction_setup
    policy_id = setup.alert_policy_id
    setup.alert_policy_id = policy_id if has_policy else None
    session.commit()
    if not fires:
        alert_forecast["A"].samples[:] = 0

    def change_policy_during_forecast(*args, **kwargs):
        setup.alert_policy_id = None if has_policy else policy_id
        session.commit()
        return alert_forecast

    prediction_id = run_stored_prediction(session, original, alert_forecast, change_policy_during_forecast)
    prediction = session.get(Prediction, prediction_id)
    assert prediction.alert_policy_id == (policy_id if has_policy else None)
    info = PredictionInfo.model_validate(prediction)
    assert info.model_dump(by_alias=True)["alertPolicyId"] == prediction.alert_policy_id
    alerts = list_alerts(session, prediction_id=prediction_id)
    assert [a.level for a in alerts] == (["action"] if has_policy and fires else [])
    assert all(a.alert_policy_id == policy_id for a in alerts)


def test_run_prediction_fails_but_keeps_prediction_when_alerts_fail(alert_prediction, alert_forecast):
    session, original = alert_prediction
    with patch("chap_core.rest_api.db_worker_functions.raise_alerts_for_prediction", side_effect=RuntimeError):
        with pytest.raises(RuntimeError):
            run_stored_prediction(session, original, alert_forecast)
    stored = session.exec(select(Prediction).where(Prediction.name == "new")).one()
    assert stored.forecasts[0].values == [21.0, 21.0]
