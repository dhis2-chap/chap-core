"""Add the installation-wide backend settings record.

Revision ID: d8e9f0a1b2c3
Revises: d5e6f7a8b9c0
"""

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import has_table

revision = "d8e9f0a1b2c3"
down_revision = "d5e6f7a8b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # create_all runs before Alembic at startup.
    if has_table("backendsettings"):
        return
    op.create_table(
        "backendsettings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("overrides", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("id = 1", name="ck_backendsettings_singleton"),
    )


def downgrade() -> None:
    op.drop_table("backendsettings")
