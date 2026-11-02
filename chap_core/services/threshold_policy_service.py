"""Service layer for ThresholdPolicy CRUD operations.

Domain logic that doesn't depend on HTTP concerns. Raises typed exceptions that
the router (or any other caller) maps to its own error format.

Only output codes can be updated: a policy's levels cannot be edited in place.
Renaming or dropping a tier would leave every existing signal naming a level the
policy no longer defines, breaking the invariant that `Signal.level` is always
one of its policy's level names. Changing the ladder requires a new policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from chap_core.database.base_tables import utc_now
from chap_core.database.signal_tables import Signal, ThresholdLevel, ThresholdPolicy, ThresholdPolicyBase
from chap_core.database.tables import Prediction, PredictionSetup

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class ThresholdPolicyServiceError(Exception):
    """Base exception for ThresholdPolicy service errors."""


class ThresholdPolicyNotFoundError(ThresholdPolicyServiceError):
    """Raised when the referenced ThresholdPolicy does not exist."""


class InvalidThresholdPolicyError(ThresholdPolicyServiceError):
    """Raised when input fails validation (empty name, no levels, duplicate level names)."""


class ThresholdPolicyInUseError(ThresholdPolicyServiceError):
    """Raised when deleting a policy that a setup, prediction or signal still points at."""


def _validate_levels(levels: Sequence[ThresholdLevel]) -> None:
    """Check that a policy has tiers with distinct, non-empty names."""
    if not levels:
        raise InvalidThresholdPolicyError("A threshold policy needs at least one level")
    names = [level.name for level in levels]
    if not all(name.strip() for name in names):
        raise InvalidThresholdPolicyError("Every threshold level needs a name")
    padded = [name for name in names if name != name.strip()]
    if padded:
        raise InvalidThresholdPolicyError(f"Threshold level names cannot start or end with whitespace, got {padded}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise InvalidThresholdPolicyError(
            f"Threshold level names must be unique within a policy, got duplicates: {duplicates}"
        )


def _validate_output_codes(levels: Sequence[ThresholdLevel], no_signal_output_code: str) -> None:
    codes = [no_signal_output_code, *(level.output_code for level in levels)]
    if any(not code or code != code.strip() for code in codes):
        raise InvalidThresholdPolicyError("Output codes must be non-empty and cannot start or end with whitespace")
    if len(set(codes)) != len(codes):
        raise InvalidThresholdPolicyError("Output codes must be unique within a policy, including the no-signal code")


def create_threshold_policy(
    session: Session,
    name: str,
    levels: Sequence[ThresholdLevel | dict[str, Any]],
    no_signal_output_code: str = "0",
) -> ThresholdPolicy:
    """Create a new ThresholdPolicy.

    Raises:
        InvalidThresholdPolicyError: name or levels fail validation.
    """
    if not name or not name.strip():
        raise InvalidThresholdPolicyError("name is required")

    try:
        validated = ThresholdPolicyBase.model_validate(
            {"name": name, "levels": levels, "no_signal_output_code": no_signal_output_code}
        )
    except ValidationError as e:
        raise InvalidThresholdPolicyError(f"Invalid threshold policy: {e}") from e
    _validate_levels(validated.levels)
    _validate_output_codes(validated.levels, validated.no_signal_output_code)

    policy = ThresholdPolicy(
        name=validated.name,
        levels=validated.levels,
        no_signal_output_code=validated.no_signal_output_code,
        created=utc_now(),
    )
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def get_threshold_policy(session: Session, policy_id: int) -> ThresholdPolicy:
    """Fetch a ThresholdPolicy by id. Raises ThresholdPolicyNotFoundError if missing."""
    policy = session.get(ThresholdPolicy, policy_id)
    if policy is None:
        raise ThresholdPolicyNotFoundError(f"ThresholdPolicy {policy_id} not found")
    return policy


def list_threshold_policies(session: Session) -> list[ThresholdPolicy]:
    return list(session.exec(select(ThresholdPolicy)).all())


def update_threshold_policy_codes(
    session: Session,
    policy_id: int,
    level_output_codes: dict[str, str],
    no_signal_output_code: str | None = None,
) -> ThresholdPolicy:
    """Update codes by level name, validating the complete mapping before saving."""
    policy = get_threshold_policy(session, policy_id)
    unknown = set(level_output_codes) - {level.name for level in policy.levels}
    if unknown:
        raise InvalidThresholdPolicyError(f"Unknown threshold levels: {sorted(unknown)}")
    levels = [
        level.model_copy(update={"output_code": level_output_codes.get(level.name, level.output_code)})
        for level in policy.levels
    ]
    no_signal_code = policy.no_signal_output_code if no_signal_output_code is None else no_signal_output_code
    _validate_output_codes(levels, no_signal_code)
    policy.levels = levels
    policy.no_signal_output_code = no_signal_code
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def delete_threshold_policy(session: Session, policy_id: int) -> None:
    """Delete a ThresholdPolicy.

    Raises:
        ThresholdPolicyNotFoundError: policy does not exist.
        ThresholdPolicyInUseError: a setup, prediction or signal still points at the
            policy. A signal's level name only means something against its policy,
            so deleting the policy would leave the signal unreadable.
    """
    policy = get_threshold_policy(session, policy_id)

    setup_id = session.exec(select(PredictionSetup.id).where(PredictionSetup.threshold_policy_id == policy_id)).first()
    if setup_id is not None:
        raise ThresholdPolicyInUseError(f"ThresholdPolicy {policy_id} is still used by prediction setup {setup_id}")

    prediction_id = session.exec(select(Prediction.id).where(Prediction.threshold_policy_id == policy_id)).first()
    if prediction_id is not None:
        raise ThresholdPolicyInUseError(f"ThresholdPolicy {policy_id} is still used by prediction {prediction_id}")

    signal_id = session.exec(select(Signal.id).where(Signal.threshold_policy_id == policy_id)).first()
    if signal_id is not None:
        raise ThresholdPolicyInUseError(
            f"ThresholdPolicy {policy_id} still has signals raised against it, e.g. {signal_id}"
        )

    session.delete(policy)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise ThresholdPolicyInUseError(f"ThresholdPolicy {policy_id} is still in use") from e
