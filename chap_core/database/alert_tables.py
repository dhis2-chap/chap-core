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

from sqlalchemy import Column
from sqlmodel import Field

from chap_core.assessment.thresholds.params import ThresholdParams
from chap_core.database.base_tables import DBModel
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
        "the threshold strategy, exactly as in `POST /v1/analytics/thresholds`.",
    )
    exceedance_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of breaching the channel, strictly above which this tier fires. "
        "`0.5` means the tier fires when more than half the forecast mass sits above the channel.",
    )

    @property
    def threshold_strategy(self) -> str:
        """Registered id of the strategy this tier's channel is drawn with."""
        return str(self.threshold_params.type)


class AlertPolicy(DBModel, table=True):
    """Persisted alert policy: a named set of alert levels a prediction setup scores against."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    name: str = Field(description="Human-friendly name for the policy.")
    created: datetime.datetime | None = Field(
        default=None, description="Server-side timestamp when the policy was created."
    )
    levels: list[AlertLevel] = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(AlertLevel)),
        description="The tiers of this policy, in the order the caller supplied them.",
    )


class AlertPolicyRead(DBModel):
    """API read shape for an `AlertPolicy`."""

    id: int = Field(description="Primary key of the policy.")
    name: str = Field(description="Human-friendly name for the policy.")
    created: datetime.datetime | None = Field(description="Server-side timestamp when the policy was created.")
    levels: list[AlertLevel] = Field(description="The tiers of this policy, in the order the caller supplied them.")
