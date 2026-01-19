"""add_archived_to_model_template

Adds archived boolean column to modeltemplatedb table for soft delete support.

Revision ID: a1b2c3d4e5f6
Revises: ff2b1bbb8418
Create Date: 2026-01-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "ff2b1bbb8418"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add archived column to modeltemplatedb table."""
    op.add_column(
        "modeltemplatedb",
        sa.Column("archived", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    """Remove archived column from modeltemplatedb table."""
    op.drop_column("modeltemplatedb", "archived")
