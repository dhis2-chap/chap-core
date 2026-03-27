"""add_uses_chapkit_to_model_template

Adds uses_chapkit boolean to modeltemplatedb so chapkit-originated templates
can be identified directly without joining through configured models.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-27

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add uses_chapkit column to modeltemplatedb table."""
    op.add_column(
        "modeltemplatedb",
        sa.Column("uses_chapkit", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    """Remove uses_chapkit column from modeltemplatedb table."""
    op.drop_column("modeltemplatedb", "uses_chapkit")
