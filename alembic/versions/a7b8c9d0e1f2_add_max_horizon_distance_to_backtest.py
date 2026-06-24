"""add_max_horizon_distance_to_backtest

Adds a nullable max_horizon_distance column to the backtest table. Existing rows
are left NULL; consumers fall back to deriving horizons from the forecasts.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-06-09

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add max_horizon_distance column to backtest table."""
    op.add_column(
        "backtest",
        sa.Column("max_horizon_distance", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """Remove max_horizon_distance column from backtest table."""
    op.drop_column("backtest", "max_horizon_distance")
