"""Service layer for Signal CRUD and the release gate.

Domain logic that doesn't depend on HTTP concerns. Raises typed exceptions that
the router (or any other caller) maps to its own error format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from chap_core.database.base_tables import utc_now
from chap_core.database.signal_tables import Signal, SignalReleaseStatus, ThresholdPolicy
from chap_core.database.tables import Prediction
from chap_core.services import threshold_service

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class SignalServiceError(Exception):
    """Base exception for Signal service errors."""


class SignalNotFoundError(SignalServiceError):
    """Raised when a referenced Signal does not exist."""


class InvalidSignalError(SignalServiceError):
    """Raised when input fails validation (unknown policy, unknown level, unknown prediction)."""


class SignalReleasedError(SignalServiceError):
    """Raised when deleting a signal that has been cleared for dissemination."""


def raise_signals_for_prediction(session: Session, prediction_id: int) -> list[Signal]:
    """Create pending signals for the most severe firing level in each forecast cell.

    Policy order defines severity. Thresholds come from the history of the
    backtest's target column, the same column the prediction forecasts.
    """
    prediction = session.get(Prediction, prediction_id)
    if prediction is None or prediction.prediction_setup is None:
        return []
    policy = prediction.threshold_policy
    if policy is None or not policy.levels:
        return []

    periods = sorted({forecast.period for forecast in prediction.forecasts})
    locations = sorted({forecast.org_unit for forecast in prediction.forecasts})
    target_column = prediction.prediction_setup.backtest.specification.target_column
    signals = {}
    for level in policy.levels:
        thresholds = threshold_service.compute_thresholds(
            session, prediction.dataset_id, periods, level.threshold_params, locations, target_column
        ).set_index(["location", "period_id"])["threshold"]
        for forecast in prediction.forecasts:
            cell = (forecast.org_unit, forecast.period)
            threshold = thresholds.get(cell, np.nan)
            if not np.isfinite(threshold):
                logger.warning(
                    "Skipping threshold level %r for prediction %s, org unit %s, period %s: no threshold",
                    level.name,
                    prediction_id,
                    *cell,
                )
                continue
            probability = np.mean(np.asarray(forecast.values) > threshold)
            if probability >= level.alarm_threshold:
                signals[cell] = Signal(
                    prediction_id=prediction_id,
                    threshold_policy_id=policy.id,
                    org_unit=forecast.org_unit,
                    time_period=forecast.period,
                    level=level.name,
                )
    return create_signals(session, list(signals.values()))


def _validate_level(session: Session, threshold_policy_id: int, level: str) -> None:
    """Check the policy exists and actually defines the named level.

    A level name only means something against its policy, so a signal naming a
    tier the policy does not have would be unreadable later.
    """
    policy = session.get(ThresholdPolicy, threshold_policy_id)
    if policy is None:
        raise InvalidSignalError(f"ThresholdPolicy {threshold_policy_id} not found")
    known = [policy_level.name for policy_level in policy.levels]
    if level not in known:
        raise InvalidSignalError(
            f"ThresholdPolicy {threshold_policy_id} has no level {level!r}; it defines {sorted(known)}"
        )


def create_signals(session: Session, signals: Sequence[Signal]) -> list[Signal]:
    """Record a batch of signals.

    Signals arrive in batches -- one run raises many at once -- so the whole batch
    is validated before anything is written and fails together.

    A prediction raises at most one signal per org unit and period, so re-running
    or re-importing it does not queue the same signal twice.

    Raises:
        InvalidSignalError: unknown policy, a level the policy does not define, an
            unknown prediction, or a second signal for the same prediction, org unit
            and period. A concurrent request can pass these checks first, so the
            database has the last word and the whole batch is rolled back.
    """
    if not signals:
        return []

    seen = set()
    for signal in signals:
        _validate_level(session, signal.threshold_policy_id, signal.level)
        if signal.prediction_id is None:
            continue
        if session.get(Prediction, signal.prediction_id) is None:
            raise InvalidSignalError(f"Prediction {signal.prediction_id} not found")
        key = (signal.prediction_id, signal.org_unit, signal.time_period)
        existing = session.exec(
            select(Signal.id).where(
                Signal.prediction_id == signal.prediction_id,
                Signal.org_unit == signal.org_unit,
                Signal.time_period == signal.time_period,
            )
        ).first()
        if key in seen or existing is not None:
            raise InvalidSignalError(
                f"Prediction {signal.prediction_id} already has a signal for {signal.org_unit} in {signal.time_period}"
            )
        seen.add(key)

    now = utc_now()
    for signal in signals:
        signal.created = now
        session.add(signal)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise InvalidSignalError("The signals conflict with a change made at the same time; none were recorded") from e
    for signal in signals:
        session.refresh(signal)
    return list(signals)


def get_signal(session: Session, signal_id: int) -> Signal:
    """Fetch a Signal by id. Raises SignalNotFoundError if missing."""
    signal = session.get(Signal, signal_id)
    if signal is None:
        raise SignalNotFoundError(f"Signal {signal_id} not found")
    return signal


def delete_signal(session: Session, signal_id: int) -> None:
    """Delete a signal, e.g. one raised by a bad run, so its prediction can be deleted.

    Raises:
        SignalNotFoundError: signal does not exist.
        SignalReleasedError: the signal has been cleared for dissemination.
    """
    signal = get_signal(session, signal_id)
    if signal.release_status is SignalReleaseStatus.RELEASED:
        raise SignalReleasedError(f"Signal {signal_id} is released for dissemination and cannot be deleted")
    session.delete(signal)
    session.commit()


def list_signals(
    session: Session,
    threshold_policy_id: int | None = None,
    prediction_id: int | None = None,
    org_unit: str | None = None,
    release_status: SignalReleaseStatus | None = None,
) -> list[Signal]:
    """List signals, narrowed by whichever filters are given.

    Filtering on ``releaseStatus=pending`` is the review queue.
    """
    statement = select(Signal)
    if threshold_policy_id is not None:
        statement = statement.where(Signal.threshold_policy_id == threshold_policy_id)
    if prediction_id is not None:
        statement = statement.where(Signal.prediction_id == prediction_id)
    if org_unit is not None:
        statement = statement.where(Signal.org_unit == org_unit)
    if release_status is not None:
        statement = statement.where(Signal.release_status == release_status)
    return list(session.exec(statement).all())


def set_release_status(
    session: Session,
    signal_ids: Sequence[int],
    release_status: SignalReleaseStatus,
    reviewed_by: str,
) -> list[Signal]:
    """Move a set of signals through the release gate together.

    Reviewers work through a queue and decide on several signals at once, so the
    whole set is applied atomically: if any id is unknown, nothing is written.

    Raises:
        SignalNotFoundError: any id does not exist.
        InvalidSignalError: no ids, or an empty reviewer.
    """
    if not signal_ids:
        raise InvalidSignalError("No signals to review")
    if not reviewed_by or not reviewed_by.strip():
        raise InvalidSignalError("reviewed_by is required")

    signals = [get_signal(session, signal_id) for signal_id in signal_ids]

    reviewed_at = utc_now() if release_status is not SignalReleaseStatus.PENDING else None
    reviewer = reviewed_by if release_status is not SignalReleaseStatus.PENDING else None
    for signal in signals:
        signal.release_status = release_status
        signal.reviewed_by = reviewer
        signal.reviewed_at = reviewed_at
        session.add(signal)
    session.commit()
    for signal in signals:
        session.refresh(signal)
    return signals
