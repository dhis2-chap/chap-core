"""add_dataset_created_manually

Flag datasets a user created directly, as opposed to those created as a side effect of
an evaluation or prediction. Existing rows are set to false: older manual and
evaluation datasets were both stored with type "evaluation" and cannot be told apart.

Revision ID: b4c5d6e7f8a3
Revises: a3b4c5d6e7f2
Create Date: 2026-09-23

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import has_column

revision: str = "b4c5d6e7f8a3"
down_revision: str | Sequence[str] | None = "a3b4c5d6e7f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COLUMN = "created_manually"


def upgrade() -> None:
    """Add the flag, defaulting existing rows to false.

    Startup runs create_all before Alembic, so the column may already exist.
    """
    if not has_column("dataset", COLUMN):
        op.add_column("dataset", sa.Column(COLUMN, sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    """Drop the flag."""
    op.drop_column("dataset", COLUMN)
