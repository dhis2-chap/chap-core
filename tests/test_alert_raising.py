"""Forecast alert selection and persistence, using real stored history and strategies."""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from chap_core.assessment.thresholds.params import SeasonalParams
from chap_core.database.alert_tables import Alert, AlertApproval
from chap_core.database.database import SessionWrapper
from chap_core.database.tables import Prediction
from chap_core.datatypes import Samples
from chap_core.rest_api.db_worker_functions import run_prediction
from chap_core.services.alert_service import list_alerts, raise_alerts_for_prediction
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange


def test_without_policy_creates_no_alerts(alert_prediction):
    session, prediction = alert_prediction
    prediction.prediction_setup.alert_policy = None
    session.commit()
    assert raise_alerts_for_prediction(session, prediction.id) == []
    assert list_alerts(session) == []


def test_most_severe_firing_level_per_cell(alert_prediction):
    session, prediction = alert_prediction
    alerts = raise_alerts_for_prediction(session, prediction.id)
    assert {(a.org_unit, a.time_period, a.level) for a in alerts} == {
        ("A", "2024-01", "monitor"),
        ("A", "2024-02", "action"),
    }
    assert all(a.approved is AlertApproval.PENDING for a in alerts)


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


def test_strategy_failure_does_not_prevent_other_levels(alert_prediction, caplog):
    session, prediction = alert_prediction
    policy = prediction.prediction_setup.alert_policy
    policy.levels = [
        policy.levels[0],
        policy.levels[1].model_copy(update={"threshold_params": SeasonalParams(type="seasonal")}),
    ]
    session.commit()
    with patch(
        "chap_core.assessment.thresholds.seasonal.SeasonalThresholdStrategy.compute", side_effect=ValueError("failed")
    ):
        with caplog.at_level(logging.WARNING):
            alerts = raise_alerts_for_prediction(session, prediction.id)
    assert [a.level for a in alerts] == ["monitor", "monitor"]
    assert "threshold computation failed" in caplog.text


def _run_prediction(session, original):
    forecast = DataSet({"A": Samples(PeriodRange.from_strings(["2024-01"]), np.array([[21.0, 21.0]]))})
    wrapper = SessionWrapper(session=session)
    with (
        patch.object(wrapper, "get_configured_model_with_code"),
        patch("chap_core.rest_api.db_worker_functions.forecast_ahead", return_value=forecast),
    ):
        return run_prediction(
            original.model_id, str(original.dataset_id), 1, "new", wrapper, original.prediction_setup_id
        )


def test_run_prediction_raises_alerts(alert_prediction):
    session, original = alert_prediction
    prediction_id = _run_prediction(session, original)
    assert [a.level for a in list_alerts(session, prediction_id=prediction_id)] == ["action"]


def test_run_prediction_keeps_prediction_when_alerts_fail(alert_prediction):
    session, original = alert_prediction

    def fail_flush(session_arg, _prediction_id):
        session_arg.add(Alert(prediction_id=-1, alert_policy_id=-1, org_unit="A", time_period="2024-01", level="x"))
        session_arg.flush()

    with patch("chap_core.rest_api.db_worker_functions.raise_alerts_for_prediction", side_effect=fail_flush):
        prediction_id = _run_prediction(session, original)

    # The session is rolled back and usable, and the prediction was committed before alerting.
    assert session.get(Prediction, prediction_id).forecasts[0].values == [21.0, 21.0]
    assert list_alerts(session, prediction_id=prediction_id) == []
