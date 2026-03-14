"""add_requires_geo_and_documentation_url

Adds requires_geo boolean to modeltemplatedb and documentation_url string
to modeltemplatedb for chapkit field mapping.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-13

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add requires_geo and documentation_url columns to modeltemplatedb."""
    op.add_column(
        "modeltemplatedb",
        sa.Column("requires_geo", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "modeltemplatedb",
        sa.Column("documentation_url", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Remove requires_geo and documentation_url columns from modeltemplatedb."""
    op.drop_column("modeltemplatedb", "documentation_url")
    op.drop_column("modeltemplatedb", "requires_geo")
