"""add_alert_policy

Creates the alertpolicy table and adds a nullable alert_policy_id foreign key
column on predictionsetup. The policy's levels live in a JSON column rather
than a table of their own: they are only ever read and written whole, with the
policy, and nothing references an individual level.

The foreign key is RESTRICT rather than SET NULL, so a policy a setup still
points at cannot be deleted out from under it.

Revision ID: c5d6e7f8a3b4
Revises: b4c5d6e7f8a3
Create Date: 2026-09-29

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from chap_core.database.migration_helpers import foreign_key_name, has_column, has_table

# revision identifiers, used by Alembic.
revision: str = "c5d6e7f8a3b4"
down_revision: Union[str, Sequence[str], None] = "b4c5d6e7f8a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not has_table("alertpolicy"):
        op.create_table(
            "alertpolicy",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("created", sa.DateTime(), nullable=True),
            sa.Column("levels", sa.JSON(), nullable=True),
        )

    if not has_column("predictionsetup", "alert_policy_id"):
        op.add_column(
            "predictionsetup",
            sa.Column("alert_policy_id", sa.Integer(), nullable=True),
        )
    if foreign_key_name("predictionsetup", "alertpolicy") is None:
        # The generic startup migration fills a new integer column with 0 on existing
        # rows, which no policy has, so clear dangling ids before the constraint
        # checks them.
        op.execute(
            sa.text(
                "UPDATE predictionsetup SET alert_policy_id = NULL "
                "WHERE alert_policy_id NOT IN (SELECT id FROM alertpolicy)"
            )
        )
        op.create_foreign_key(
            "fk_predictionsetup_alert_policy",
            "predictionsetup",
            "alertpolicy",
            ["alert_policy_id"],
            ["id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    foreign_key = foreign_key_name("predictionsetup", "alertpolicy")
    if foreign_key is not None:
        op.drop_constraint(foreign_key, "predictionsetup", type_="foreignkey")
    op.drop_column("predictionsetup", "alert_policy_id")
    op.drop_table("alertpolicy")
