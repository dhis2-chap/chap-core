"""add_dataset_masked_ranges

Store the inclusive period ranges in which a dataset's `disease_cases` is masked from
models (CLIM-1300). Existing datasets get no ranges, so they behave as before.

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-10-22

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import has_column

revision: str = "e6f7a8b9c0d1"
down_revision: str | Sequence[str] | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COLUMN = "masked_ranges"


def upgrade() -> None:
    """Add the column, empty for existing rows.

    Startup runs create_all before Alembic, so the column may already exist.
    """
    if not has_column("dataset", COLUMN):
        op.add_column("dataset", sa.Column(COLUMN, sa.JSON(), nullable=True))


def downgrade() -> None:
    """Drop the column."""
    op.drop_column("dataset", COLUMN)
