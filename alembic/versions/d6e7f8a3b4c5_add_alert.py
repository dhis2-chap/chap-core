"""add_alert

Creates the alert table: one alert per (org unit, period) that breached a level
of an AlertPolicy, plus the release gate recording whether a human has cleared
it for dissemination.

The gate is stored as a plain string rather than a native enum, so adding a
state later needs no type migration.

The policy foreign key is RESTRICT: an alert's level name only means something
against its policy, so the policy cannot be deleted out from under it. The
prediction foreign key is SET NULL, so deleting a run does not erase the record
of what it raised.

Revision ID: d6e7f8a3b4c5
Revises: c5d6e7f8a3b4
Create Date: 2026-09-29

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from chap_core.database.migration_helpers import has_table

# revision identifiers, used by Alembic.
revision: str = "d6e7f8a3b4c5"
down_revision: Union[str, Sequence[str], None] = "c5d6e7f8a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not has_table("alert"):
        op.create_table(
            "alert",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("created", sa.DateTime(), nullable=True),
            sa.Column("time_period", sa.String(), nullable=False),
            sa.Column("org_unit", sa.String(), nullable=False),
            sa.Column("alert_policy_id", sa.Integer(), nullable=False),
            sa.Column("level", sa.String(), nullable=False),
            sa.Column("prediction_id", sa.Integer(), nullable=True),
            sa.Column("approved", sa.String(), nullable=False, server_default="pending"),
            sa.Column("approved_by", sa.String(), nullable=True),
            sa.Column("approved_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(
                ["alert_policy_id"],
                ["alertpolicy.id"],
                ondelete="RESTRICT",
                name="fk_alert_alert_policy",
            ),
            sa.ForeignKeyConstraint(
                ["prediction_id"],
                ["prediction.id"],
                ondelete="SET NULL",
                name="fk_alert_prediction",
            ),
        )
        op.create_index("ix_alert_alert_policy_id", "alert", ["alert_policy_id"])
        op.create_index("ix_alert_approved", "alert", ["approved"])


def downgrade() -> None:
    op.drop_table("alert")
