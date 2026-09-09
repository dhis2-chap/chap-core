"""add_backtest_future_weather_provider

Store which future-weather provider supplied the climate covariates a backtest
ran with, on the backtest row.

Existing rows are backfilled with "climatology". Every backtest row was written
by the REST path, which hardcoded QuickForecastFetcher - a per-location seasonal
regression fitted on the historical data, which is exactly what the climatology
provider does. Note this differs from evaluation .nc files, which came from the
CLI and were given the forecast window's observed weather; those are read back
as "observed" instead.

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-09-08

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = "d0e1f2a3b4c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COLUMN = "future_weather_provider"
# The provider the REST backtest path always used. Local to this migration so it
# stays fixed even if the registry default changes later.
LEGACY_PROVIDER = "climatology"


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col["name"] == column for col in inspector.get_columns(table))


def upgrade() -> None:
    """Add the provider column, backfill existing rows, then make it NOT NULL.

    Startup runs a generic metadata migration before Alembic, so the column may
    already exist with an empty value in every row; both shapes are backfilled
    the same way.
    """
    if not _has_column("backtest", COLUMN):
        op.add_column("backtest", sa.Column(COLUMN, sa.String(), nullable=True))
    op.execute(
        sa.text(f"UPDATE backtest SET {COLUMN} = :provider WHERE {COLUMN} IS NULL OR {COLUMN} = ''").bindparams(
            provider=LEGACY_PROVIDER
        )
    )
    op.alter_column("backtest", COLUMN, existing_type=sa.String(), nullable=False)


def downgrade() -> None:
    """Drop the provider column."""
    op.drop_column("backtest", COLUMN)
