"""Add specificationbaseline, recording each baseline model's run on a backtest specification.

A row claims the run of one baseline configured model on one specification, so the
baseline is queued once however many triggers ask for it. Existing specifications get no
rows here: their baselines are queued the first time the specification is read.

Revision ID: c4d5e6f7a8b9
Revises: de7478f3ff6d
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import has_table

revision: str = "c4d5e6f7a8b9"
down_revision: str | Sequence[str] | None = "de7478f3ff6d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "specificationbaseline"


def upgrade() -> None:
    # create_all runs before Alembic, so the table is usually already there.
    if has_table(TABLE):
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("specification_id", sa.Integer(), nullable=False),
        sa.Column("configured_model_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("backtest_id", sa.Integer(), nullable=True),
        sa.Column("message", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["specification_id"], ["backtestspecification.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["configured_model_id"], ["configuredmodeldb.id"]),
        sa.ForeignKeyConstraint(["backtest_id"], ["backtest.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("specification_id", "configured_model_id", name="uq_specificationbaseline_model"),
    )


def downgrade() -> None:
    op.drop_table(TABLE)
