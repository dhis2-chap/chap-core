"""drop_debugentry_table

Drops the debugentry table now that the debug endpoints have been removed
and replaced by a synchronous /health/ready readiness check.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop debugentry table if it exists."""
    op.execute("DROP TABLE IF EXISTS debugentry")


def downgrade() -> None:
    """Recreate debugentry table."""
    op.create_table(
        "debugentry",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
