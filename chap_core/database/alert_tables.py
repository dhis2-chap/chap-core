"""Alert policy tables: the tiered channels alerts are raised against.

An :class:`AlertPolicy` is a named ladder of :class:`AlertLevel` tiers -- monitor,
alert, action and so on -- each pairing an epidemic channel with the exceedance
probability at which that tier fires. A :class:`~chap_core.database.tables.PredictionSetup`
points at one policy, so a recurring forecast knows what counts as an alert.

The levels are stored as JSON on the policy row rather than as a table of their
own: they are only ever read and written whole, with the policy, and nothing
references an individual level.

A level names its strategy through ``threshold_params.type``, the same
discriminator the thresholds API uses to select a strategy, so there is no
second field that could disagree with it.
"""

import datetime
from enum import StrEnum

from pydantic import field_validator
from sqlalchemy import Column, String, TypeDecorator, UniqueConstraint
from sqlmodel import Field

from chap_core.assessment.thresholds.params import ThresholdParams
from chap_core.database.base_tables import DBModel, PeriodID
from chap_core.database.dataset_tables import PydanticListType


class AlertLevel(DBModel):
    """One tier of an :class:`AlertPolicy`.

    Stored inside the policy's JSON column, never as a row of its own.
    """

    name: str = Field(
        description="Name of the tier, e.g. `monitor`, `alert` or `action`. Unique within a policy.",
    )
    threshold_params: ThresholdParams = Field(
        description="Parameters for the epidemic channel this tier is judged against. Its `type` selects "
        "the threshold strategy, exactly as in `POST /v1/analytics/thresholds`. Must describe a single "
        "threshold line, so it is unambiguous which line the tier is judged against.",
    )
    exceedance_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of breaching the channel at or above which this tier fires. "
        "`0.5` means the tier fires when at least half the forecast mass sits above the channel.",
    )

    @field_validator("threshold_params")
    @classmethod
    def _single_line(cls, params: ThresholdParams) -> ThresholdParams:
        if len(params.lines) != 1:
            raise ValueError(f"An alert level is judged against one threshold line, got {params.lines}")
        return params

    @property
    def threshold_strategy(self) -> str:
        """Registered id of the strategy this tier's channel is drawn with."""
        return str(self.threshold_params.type)


class AlertPolicyBase(DBModel):
    """Fields shared by every alert-policy shape (DB row, read view, create body)."""

    name: str = Field(description="Human-friendly name for the policy.")
    levels: list[AlertLevel] = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(AlertLevel)),
        description="The tiers of this policy, ordered from least to most severe.",
    )


class AlertPolicy(AlertPolicyBase, table=True):
    """Persisted alert policy: a named set of alert levels a prediction setup scores against."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    created: datetime.datetime | None = Field(
        default=None, description="Server-side timestamp when the policy was created."
    )


class AlertPolicyRead(AlertPolicyBase):
    """API read shape for an `AlertPolicy`."""

    id: int = Field(description="Primary key of the policy.")
    created: datetime.datetime | None = Field(description="Server-side timestamp when the policy was created.")


class AlertApproval(StrEnum):
    """Where an alert stands in the release gate.

    This is about dissemination, not epidemiology: it records whether a human has
    cleared the alert to go out to recipients, and says nothing about whether the
    outbreak turned out to be real.
    """

    PENDING = "pending"
    """Nobody has reviewed it yet. This is the review queue."""

    APPROVED = "approved"
    """Cleared for release; dissemination picks these up."""

    DECLINED = "declined"
    """Reviewed and deliberately not released. Distinct from `PENDING`, so a
    declined alert does not come back around for re-triage."""


class AlertApprovalType(TypeDecorator):
    """Stores :class:`AlertApproval` as an unbounded string, reading it back as the enum.

    A plain string column rather than a database enum type or a length-bounded
    one, so adding a state later needs no schema migration. The decorator is what
    keeps the round trip honest: without it a loaded row hands back a bare `str`
    while the model claims an `AlertApproval`, and an `is` comparison silently fails.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return AlertApproval(value).value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return AlertApproval(value)


class AlertBase(DBModel):
    """What an alert is: which policy level fired, and where and when.

    Also the create shape. The release-gate fields live on :class:`AlertRecord`
    instead, so a caller cannot record an alert that is already cleared for
    dissemination -- every alert starts in `pending`.
    """

    time_period: PeriodID = Field(description="Period the alert is about, e.g. `2024-07`.")
    org_unit: str = Field(description="Identifier of the org unit the alert is for.")
    alert_policy_id: int = Field(
        foreign_key="alertpolicy.id",
        ondelete="RESTRICT",
        index=True,
        description="Foreign key to the `AlertPolicy` whose level was breached.",
    )
    level: str = Field(
        description="Name of the `AlertLevel` that fired, e.g. `monitor` or `action`. "
        "Always one of the names on the referenced policy.",
    )
    prediction_id: int | None = Field(
        default=None,
        foreign_key="prediction.id",
        ondelete="RESTRICT",
        description="Foreign key to the `Prediction` whose forecast raised the alert; `None` if it "
        "was recorded without a linked run. Deleting a prediction that raised alerts is refused, so "
        "an alert is never separated from the forecast that justified it.",
    )


class AlertRecord(AlertBase):
    """A recorded alert: identity plus everything the server owns, minus the key.

    Shared by the DB row and the read view, which differ only in how `id` is typed.
    """

    created: datetime.datetime | None = Field(
        default=None, description="Server-side timestamp when the alert was recorded."
    )
    approved: AlertApproval = Field(
        default=AlertApproval.PENDING,
        sa_column=Column(
            AlertApprovalType(),
            nullable=False,
            server_default=AlertApproval.PENDING.value,
            index=True,
        ),
        description="Where the alert stands in the release gate. Cleared for dissemination only when "
        "`approved`; says nothing about whether the outbreak was real.",
    )
    approved_by: str | None = Field(
        default=None, description="Caller-supplied identifier of whoever reviewed it; `None` while pending."
    )
    approved_at: datetime.datetime | None = Field(
        default=None, description="Server-side timestamp of the review; `None` while pending."
    )


class Alert(AlertRecord, table=True):
    """One alert raised for a `(location, period)` against a level of an `AlertPolicy`.

    A prediction raises at most one alert per org unit and period, carrying the most
    severe level breached.
    """

    __table_args__ = (
        UniqueConstraint("prediction_id", "org_unit", "time_period", name="uq_alert_prediction_org_unit_period"),
    )

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")


class AlertRead(AlertRecord):
    """API read shape for an `Alert`.

    Carries the policy's id rather than the policy itself: a listing is usually long
    and the caller already has the policies.
    """

    id: int = Field(description="Primary key of the alert.")
