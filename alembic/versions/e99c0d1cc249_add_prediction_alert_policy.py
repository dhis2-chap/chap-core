"""add_prediction_alert_policy

Revision ID: e99c0d1cc249
Revises: d8e9f0a1b2c3
Create Date: 2026-10-27 20:37:00.468523

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import foreign_key_name, has_column

# revision identifiers, used by Alembic.
revision: str = "e99c0d1cc249"
down_revision: str | Sequence[str] | None = "d8e9f0a1b2c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if not has_column("prediction", "alert_policy_id"):
        op.add_column("prediction", sa.Column("alert_policy_id", sa.Integer(), nullable=True))
    if foreign_key_name("prediction", "alertpolicy") is None:
        # Startup may already have added this column with 0 on existing rows.
        # No historical policy was recorded, so leave all those rows unknown.
        op.execute("UPDATE prediction SET alert_policy_id = NULL")
        op.create_foreign_key(
            "fk_prediction_alert_policy",
            "prediction",
            "alertpolicy",
            ["alert_policy_id"],
            ["id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    foreign_key = foreign_key_name("prediction", "alertpolicy")
    if foreign_key is not None:
        op.drop_constraint(foreign_key, "prediction", type_="foreignkey")
    op.drop_column("prediction", "alert_policy_id")
