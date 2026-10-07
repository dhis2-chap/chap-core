"""Service layer for Alert CRUD and the release gate.

Domain logic that doesn't depend on HTTP concerns. Raises typed exceptions that
the router (or any other caller) maps to its own error format.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, select

from chap_core.database.alert_tables import Alert, AlertApproval, AlertPolicy
from chap_core.database.tables import Prediction
from chap_core.services import threshold_service

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class AlertServiceError(Exception):
    """Base exception for Alert service errors."""


class AlertNotFoundError(AlertServiceError):
    """Raised when a referenced Alert does not exist."""


class InvalidAlertError(AlertServiceError):
    """Raised when input fails validation (unknown policy, unknown level, unknown prediction)."""


class AlertApprovedError(AlertServiceError):
    """Raised when deleting an alert that has been cleared for dissemination."""


def raise_alerts_for_prediction(session: Session, prediction_id: int) -> list[Alert]:
    """Create pending alerts for the most severe firing level in each forecast cell.

    Policy order defines severity.
    """
    prediction = session.get(Prediction, prediction_id)
    if prediction is None or prediction.prediction_setup is None:
        return []
    policy = prediction.prediction_setup.alert_policy
    if policy is None or not policy.levels:
        return []

    periods = sorted({forecast.period for forecast in prediction.forecasts})
    locations = sorted({forecast.org_unit for forecast in prediction.forecasts})
    alerts = {}
    for level in policy.levels:
        try:
            thresholds = threshold_service.compute_thresholds(
                session, prediction.dataset_id, periods, level.threshold_params, locations
            ).set_index(["location", "period_id"])["threshold"]
        except (threshold_service.NoObservationsError, threshold_service.InvalidThresholdInputError):
            logger.warning(
                "Skipping alert level %r for prediction %s: threshold computation failed",
                level.name,
                prediction_id,
                exc_info=True,
            )
            continue

        for forecast in prediction.forecasts:
            cell = (forecast.org_unit, forecast.period)
            threshold = thresholds.get(cell, np.nan)
            if not np.isfinite(threshold):
                logger.warning(
                    "Skipping alert level %r for prediction %s, org unit %s, period %s: no threshold",
                    level.name,
                    prediction_id,
                    *cell,
                )
                continue
            probability = np.mean(np.asarray(forecast.values) > threshold)
            if probability >= level.exceedance_threshold:
                alerts[cell] = Alert(
                    prediction_id=prediction_id,
                    alert_policy_id=policy.id,
                    org_unit=forecast.org_unit,
                    time_period=forecast.period,
                    level=level.name,
                )
    return create_alerts(session, list(alerts.values()))


def _validate_level(session: Session, alert_policy_id: int, level: str) -> None:
    """Check the policy exists and actually defines the named level.

    A level name only means something against its policy, so an alert naming a
    tier the policy does not have would be unreadable later.
    """
    policy = session.get(AlertPolicy, alert_policy_id)
    if policy is None:
        raise InvalidAlertError(f"AlertPolicy {alert_policy_id} not found")
    known = [policy_level.name for policy_level in policy.levels]
    if level not in known:
        raise InvalidAlertError(f"AlertPolicy {alert_policy_id} has no level {level!r}; it defines {sorted(known)}")


def create_alerts(session: Session, alerts: Sequence[Alert]) -> list[Alert]:
    """Record a batch of alerts.

    Alerts arrive in batches -- one run raises many at once -- so the whole batch
    is validated before anything is written and fails together.

    A prediction raises at most one alert per org unit and period, so re-running
    or re-importing it does not queue the same alert twice.

    Raises:
        InvalidAlertError: unknown policy, a level the policy does not define, an
            unknown prediction, or a second alert for the same prediction, org unit
            and period.
    """
    if not alerts:
        return []

    seen = set()
    for alert in alerts:
        _validate_level(session, alert.alert_policy_id, alert.level)
        if alert.prediction_id is None:
            continue
        if session.get(Prediction, alert.prediction_id) is None:
            raise InvalidAlertError(f"Prediction {alert.prediction_id} not found")
        key = (alert.prediction_id, alert.org_unit, alert.time_period)
        existing = session.exec(
            select(Alert.id).where(
                Alert.prediction_id == alert.prediction_id,
                Alert.org_unit == alert.org_unit,
                Alert.time_period == alert.time_period,
            )
        ).first()
        if key in seen or existing is not None:
            raise InvalidAlertError(
                f"Prediction {alert.prediction_id} already has an alert for {alert.org_unit} in {alert.time_period}"
            )
        seen.add(key)

    now = datetime.datetime.now()
    for alert in alerts:
        alert.created = now
        session.add(alert)
    session.commit()
    for alert in alerts:
        session.refresh(alert)
    return list(alerts)


def get_alert(session: Session, alert_id: int) -> Alert:
    """Fetch an Alert by id. Raises AlertNotFoundError if missing."""
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise AlertNotFoundError(f"Alert {alert_id} not found")
    return alert


def delete_alert(session: Session, alert_id: int) -> None:
    """Delete an alert, e.g. one raised by a bad run, so its prediction can be deleted.

    Raises:
        AlertNotFoundError: alert does not exist.
        AlertApprovedError: the alert has been cleared for dissemination.
    """
    alert = get_alert(session, alert_id)
    if alert.approved is AlertApproval.APPROVED:
        raise AlertApprovedError(f"Alert {alert_id} is approved for dissemination and cannot be deleted")
    session.delete(alert)
    session.commit()


def list_alerts(
    session: Session,
    alert_policy_id: int | None = None,
    prediction_id: int | None = None,
    org_unit: str | None = None,
    approved: AlertApproval | None = None,
) -> list[Alert]:
    """List alerts, narrowed by whichever filters are given.

    Filtering on ``approved=pending`` is the review queue.
    """
    statement = select(Alert)
    if alert_policy_id is not None:
        statement = statement.where(Alert.alert_policy_id == alert_policy_id)
    if prediction_id is not None:
        statement = statement.where(Alert.prediction_id == prediction_id)
    if org_unit is not None:
        statement = statement.where(Alert.org_unit == org_unit)
    if approved is not None:
        statement = statement.where(Alert.approved == approved)
    return list(session.exec(statement).all())


def set_approval(
    session: Session,
    alert_ids: Sequence[int],
    approved: AlertApproval,
    approved_by: str,
) -> list[Alert]:
    """Move a set of alerts through the release gate together.

    Reviewers work through a queue and decide on several alerts at once, so the
    whole set is applied atomically: if any id is unknown, nothing is written.

    Raises:
        AlertNotFoundError: any id does not exist.
        InvalidAlertError: no ids, or an empty reviewer.
    """
    if not alert_ids:
        raise InvalidAlertError("No alerts to review")
    if not approved_by or not approved_by.strip():
        raise InvalidAlertError("approved_by is required")

    alerts = [get_alert(session, alert_id) for alert_id in alert_ids]

    reviewed_at = datetime.datetime.now() if approved is not AlertApproval.PENDING else None
    reviewer = approved_by if approved is not AlertApproval.PENDING else None
    for alert in alerts:
        alert.approved = approved
        alert.approved_by = reviewer
        alert.approved_at = reviewed_at
        session.add(alert)
    session.commit()
    for alert in alerts:
        session.refresh(alert)
    return alerts
