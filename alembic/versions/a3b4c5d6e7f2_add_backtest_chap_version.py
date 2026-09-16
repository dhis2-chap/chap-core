"""add_backtest_chap_version

Record which chap-core release produced each backtest, so a metric shift over time
can be attributed to the model or to the platform. Existing rows are left NULL: the
version that produced them was never recorded and cannot be recovered.

Revision ID: a3b4c5d6e7f2
Revises: f2a3b4c5d6e1
Create Date: 2026-09-16

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a3b4c5d6e7f2"
down_revision: str | Sequence[str] | None = "f2a3b4c5d6e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COLUMN = "chap_version"


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col["name"] == column for col in inspector.get_columns(table))


def upgrade() -> None:
    """Add the nullable version column, without backfill.

    Startup runs create_all before Alembic, so the column may already exist.
    """
    if not _has_column("backtest", COLUMN):
        op.add_column("backtest", sa.Column(COLUMN, sa.String(), nullable=True))


def downgrade() -> None:
    """Drop the version column."""
    op.drop_column("backtest", COLUMN)
