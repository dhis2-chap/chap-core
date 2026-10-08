"""Add a role to model templates, marking baseline and comparison models.

Revision ID: de7478f3ff6d
Revises: e7f8a3b4c5d6
"""

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import has_column

revision = "de7478f3ff6d"
down_revision = "e7f8a3b4c5d6"
branch_labels = None
depends_on = None

TABLE = "modeltemplatedb"


def upgrade() -> None:
    if not has_column(TABLE, "role"):
        op.add_column(TABLE, sa.Column("role", sa.String(length=32), nullable=True))
    # The generic startup migration may have already added the column as an empty string.
    op.execute("UPDATE modeltemplatedb SET role = NULL WHERE role = ''")


def downgrade() -> None:
    op.drop_column(TABLE, "role")
