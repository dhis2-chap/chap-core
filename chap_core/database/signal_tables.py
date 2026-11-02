"""Forecast signals and the threshold policies that generate them.

A signal is unverified. Releasing it for dissemination does not verify an event
or assess its public-health risk; see docs/contributor/vocabulary.md.

A :class:`ThresholdPolicy` is a named ladder of :class:`ThresholdLevel` tiers,
each pairing an outbreak threshold with the exceedance probability at which
that tier fires. A :class:`~chap_core.database.tables.PredictionSetup`
points at one policy, so a recurring forecast knows what counts as a signal.

The levels are stored as JSON on the policy row rather than as a table of their
own: they are only ever read and written whole, with the policy, and nothing
references an individual level.

A level names its strategy through ``threshold_params.type``, the same
discriminator the thresholds API uses to select a strategy, so there is no
second field that could disagree with it.
"""

from enum import StrEnum

from pydantic import field_validator
from sqlalchemy import Column, String, TypeDecorator, UniqueConstraint
from sqlmodel import Field

from chap_core.assessment.thresholds.params import ThresholdParams
from chap_core.database.base_tables import DBModel, PeriodID, UtcDatetime
from chap_core.database.dataset_tables import PydanticListType


class ThresholdLevel(DBModel):
    """One tier of a :class:`ThresholdPolicy`.

    Stored inside the policy's JSON column, never as a row of its own.
    """

    name: str = Field(
        description="Name of the tier, e.g. `level_1` or `level_2`. Unique within a policy.",
    )
    output_code: str = Field(
        min_length=1,
        description="Output code for this level. Must be non-empty and unique within the policy, including "
        "the no-signal code. For DHIS2 import, it must match the destination data element's value type or option set.",
    )
    threshold_params: ThresholdParams = Field(
        description="Parameters for the outbreak threshold this tier is judged against. Its `type` selects "
        "the threshold strategy, exactly as in `POST /v1/analytics/thresholds`. Must describe a single "
        "threshold line, so it is unambiguous which line the tier is judged against.",
    )
    alarm_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of breaching the channel at or above which this tier fires. "
        "`0.5` means the tier fires when at least half the forecast mass sits above the channel.",
    )

    @field_validator("threshold_params")
    @classmethod
    def _single_line(cls, params: ThresholdParams) -> ThresholdParams:
        if len(params.lines) != 1:
            raise ValueError(f"A threshold level is judged against one threshold line, got {params.lines}")
        return params

    @property
    def threshold_strategy(self) -> str:
        """Registered id of the strategy this tier's channel is drawn with."""
        return str(self.threshold_params.type)


class ThresholdPolicyBase(DBModel):
    """Fields shared by every threshold-policy shape (DB row, read view, create body)."""

    name: str = Field(description="Human-friendly name for the policy.")
    no_signal_output_code: str = Field(
        default="0",
        min_length=1,
        description="Output code for the no-signal state. Must be non-empty and distinct from every level's code. "
        "For DHIS2 import, it must match the destination data element's value type or option set.",
    )
    levels: list[ThresholdLevel] = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(ThresholdLevel)),
        description="The tiers of this policy, ordered from least to most severe.",
    )

    @field_validator("levels", mode="before")
    @classmethod
    def _default_output_codes(cls, levels):
        if not isinstance(levels, (list, tuple)):
            return levels
        return [
            {"output_code": str(index), **level} if isinstance(level, dict) and "outputCode" not in level else level
            for index, level in enumerate(levels, start=1)
        ]


class ThresholdPolicy(ThresholdPolicyBase, table=True):
    """Persisted threshold policy: a named set of threshold levels a prediction setup scores against."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    created: UtcDatetime | None = Field(default=None, description="Server-side timestamp when the policy was created.")


class ThresholdPolicyRead(ThresholdPolicyBase):
    """API read shape for a `ThresholdPolicy`."""

    id: int = Field(description="Primary key of the policy.")
    created: UtcDatetime | None = Field(description="Server-side timestamp when the policy was created.")


class SignalReleaseStatus(StrEnum):
    """Where a signal stands in the release gate.

    This is about dissemination, not epidemiology: it records whether a human has
    cleared the signal to go out to recipients, and says nothing about whether the
    outbreak turned out to be real.
    """

    PENDING = "pending"
    """Nobody has reviewed it yet. This is the review queue."""

    RELEASED = "released"
    """Cleared for release; dissemination picks these up."""

    WITHHELD = "withheld"
    """Reviewed and deliberately not released. Distinct from `PENDING`, so a
    withheld signal does not come back around for another release review."""


class SignalReleaseStatusType(TypeDecorator):
    """Stores :class:`SignalReleaseStatus` as an unbounded string, reading it back as the enum.

    A plain string column rather than a database enum type or a length-bounded
    one, so adding a state later needs no schema migration. The decorator is what
    keeps the round trip honest: without it a loaded row hands back a bare `str`
    while the model claims a `SignalReleaseStatus`, and an `is` comparison silently fails.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return SignalReleaseStatus(value).value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return SignalReleaseStatus(value)


class SignalBase(DBModel):
    """What a signal is: which policy level fired, and where and when.

    Also the create shape. The release-gate fields live on :class:`SignalRecord`
    instead, so a caller cannot record a signal that is already cleared for
    dissemination -- every signal starts in `pending`.
    """

    time_period: PeriodID = Field(description="Period the signal is about, e.g. `2024-07`.")
    org_unit: str = Field(description="Identifier of the org unit the signal is for.")
    threshold_policy_id: int = Field(
        foreign_key="thresholdpolicy.id",
        ondelete="RESTRICT",
        index=True,
        description="Foreign key to the `ThresholdPolicy` whose level was breached.",
    )
    level: str = Field(
        description="Name of the `ThresholdLevel` that fired, e.g. `level_1` or `level_2`. "
        "Always one of the names on the referenced policy.",
    )
    prediction_id: int | None = Field(
        default=None,
        foreign_key="prediction.id",
        ondelete="RESTRICT",
        description="Foreign key to the `Prediction` whose forecast raised the signal; `None` if it "
        "was recorded without a linked run. Deleting a prediction that raised signals is refused, so "
        "a signal is never separated from the forecast that justified it.",
    )


class SignalRecord(SignalBase):
    """A recorded signal: identity plus everything the server owns, minus the key.

    Shared by the DB row and the read view, which differ only in how `id` is typed.
    """

    created: UtcDatetime | None = Field(default=None, description="Server-side timestamp when the signal was recorded.")
    release_status: SignalReleaseStatus = Field(
        default=SignalReleaseStatus.PENDING,
        sa_column=Column(
            SignalReleaseStatusType(),
            nullable=False,
            server_default=SignalReleaseStatus.PENDING.value,
            index=True,
        ),
        description="Where the signal stands in the release gate. Cleared for dissemination only when "
        "`released`; says nothing about whether the outbreak was real.",
    )
    reviewed_by: str | None = Field(
        default=None, description="Caller-supplied identifier of whoever reviewed it; `None` while pending."
    )
    reviewed_at: UtcDatetime | None = Field(
        default=None, description="Server-side timestamp of the review; `None` while pending."
    )


class Signal(SignalRecord, table=True):
    """One signal raised for a `(location, period)` against a level of a `ThresholdPolicy`.

    A prediction raises at most one signal per org unit and period, carrying the most
    severe level breached.
    """

    __table_args__ = (
        UniqueConstraint("prediction_id", "org_unit", "time_period", name="uq_signal_prediction_org_unit_period"),
    )

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")


class SignalRead(SignalRecord):
    """API read shape for a `Signal`.

    Carries the policy's id rather than the policy itself: a listing is usually long
    and the caller already has the policies.
    """

    id: int = Field(description="Primary key of the signal.")
