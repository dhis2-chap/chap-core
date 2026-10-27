"""Service layer for AlertPolicy CRUD operations.

Domain logic that doesn't depend on HTTP concerns. Raises typed exceptions that
the router (or any other caller) maps to its own error format.

Only output codes can be updated: a policy's levels cannot be edited in place.
Renaming or dropping a tier would leave every existing alert naming a level the
policy no longer defines, breaking the invariant that `Alert.level` is always
one of its policy's level names. Changing the ladder requires a new policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from chap_core.database.alert_tables import Alert, AlertLevel, AlertPolicy, AlertPolicyBase
from chap_core.database.base_tables import utc_now
from chap_core.database.tables import Prediction, PredictionSetup

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class AlertPolicyServiceError(Exception):
    """Base exception for AlertPolicy service errors."""


class AlertPolicyNotFoundError(AlertPolicyServiceError):
    """Raised when the referenced AlertPolicy does not exist."""


class InvalidAlertPolicyError(AlertPolicyServiceError):
    """Raised when input fails validation (empty name, no levels, duplicate level names)."""


class AlertPolicyInUseError(AlertPolicyServiceError):
    """Raised when deleting a policy that a setup, prediction or alert still points at."""


def _validate_levels(levels: Sequence[AlertLevel]) -> None:
    """Check that a policy has tiers with distinct, non-empty names."""
    if not levels:
        raise InvalidAlertPolicyError("An alert policy needs at least one level")
    names = [level.name for level in levels]
    if not all(name.strip() for name in names):
        raise InvalidAlertPolicyError("Every alert level needs a name")
    padded = [name for name in names if name != name.strip()]
    if padded:
        raise InvalidAlertPolicyError(f"Alert level names cannot start or end with whitespace, got {padded}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise InvalidAlertPolicyError(f"Alert level names must be unique within a policy, got duplicates: {duplicates}")


def _validate_output_codes(levels: Sequence[AlertLevel], no_alert_output_code: str) -> None:
    codes = [no_alert_output_code, *(level.output_code for level in levels)]
    if any(not code or code != code.strip() for code in codes):
        raise InvalidAlertPolicyError("Output codes must be non-empty and cannot start or end with whitespace")
    if len(set(codes)) != len(codes):
        raise InvalidAlertPolicyError("Output codes must be unique within a policy, including the no-alert code")


def create_alert_policy(
    session: Session,
    name: str,
    levels: Sequence[AlertLevel | dict[str, Any]],
    no_alert_output_code: str = "0",
) -> AlertPolicy:
    """Create a new AlertPolicy.

    Raises:
        InvalidAlertPolicyError: name or levels fail validation.
    """
    if not name or not name.strip():
        raise InvalidAlertPolicyError("name is required")

    try:
        validated = AlertPolicyBase.model_validate(
            {"name": name, "levels": levels, "no_alert_output_code": no_alert_output_code}
        )
    except ValidationError as e:
        raise InvalidAlertPolicyError(f"Invalid alert policy: {e}") from e
    _validate_levels(validated.levels)
    _validate_output_codes(validated.levels, validated.no_alert_output_code)

    policy = AlertPolicy(
        name=validated.name,
        levels=validated.levels,
        no_alert_output_code=validated.no_alert_output_code,
        created=utc_now(),
    )
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def get_alert_policy(session: Session, policy_id: int) -> AlertPolicy:
    """Fetch an AlertPolicy by id. Raises AlertPolicyNotFoundError if missing."""
    policy = session.get(AlertPolicy, policy_id)
    if policy is None:
        raise AlertPolicyNotFoundError(f"AlertPolicy {policy_id} not found")
    return policy


def list_alert_policies(session: Session) -> list[AlertPolicy]:
    return list(session.exec(select(AlertPolicy)).all())


def update_alert_policy_codes(
    session: Session,
    policy_id: int,
    level_output_codes: dict[str, str],
    no_alert_output_code: str | None = None,
) -> AlertPolicy:
    """Update codes by level name, validating the complete mapping before saving."""
    policy = get_alert_policy(session, policy_id)
    unknown = set(level_output_codes) - {level.name for level in policy.levels}
    if unknown:
        raise InvalidAlertPolicyError(f"Unknown alert levels: {sorted(unknown)}")
    levels = [
        level.model_copy(update={"output_code": level_output_codes.get(level.name, level.output_code)})
        for level in policy.levels
    ]
    no_alert_code = policy.no_alert_output_code if no_alert_output_code is None else no_alert_output_code
    _validate_output_codes(levels, no_alert_code)
    policy.levels = levels
    policy.no_alert_output_code = no_alert_code
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def delete_alert_policy(session: Session, policy_id: int) -> None:
    """Delete an AlertPolicy.

    Raises:
        AlertPolicyNotFoundError: policy does not exist.
        AlertPolicyInUseError: a setup, prediction or alert still points at the policy.
    """
    policy = get_alert_policy(session, policy_id)

    setup_id = session.exec(select(PredictionSetup.id).where(PredictionSetup.alert_policy_id == policy_id)).first()
    if setup_id is not None:
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} is still used by prediction setup {setup_id}")

    prediction_id = session.exec(select(Prediction.id).where(Prediction.alert_policy_id == policy_id)).first()
    if prediction_id is not None:
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} is still used by prediction {prediction_id}")

    alert_id = session.exec(select(Alert.id).where(Alert.alert_policy_id == policy_id)).first()
    if alert_id is not None:
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} still has alerts raised against it, e.g. {alert_id}")

    session.delete(policy)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} is still in use") from e
