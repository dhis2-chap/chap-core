"""Add the dataset target chosen at creation (CLIM-1466).

Revision ID: f0a1b2c3d4e5
Revises: e99c0d1cc249
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import has_column

revision: str = "f0a1b2c3d4e5"
down_revision: str | Sequence[str] | None = "e99c0d1cc249"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if not has_column("dataset", "target_column"):
        op.add_column("dataset", sa.Column("target_column", sa.String(), nullable=True))
    # Startup can already have added the column and filled it with the default, so every existing
    # value is a placeholder. A dataset whose backtests all evaluated one target keeps that target.
    op.execute(
        "UPDATE dataset SET target_column = spec.target_column FROM ("
        "SELECT dataset_id, min(target_column) AS target_column FROM backtestspecification "
        "GROUP BY dataset_id HAVING count(DISTINCT target_column) = 1) spec "
        "WHERE dataset.id = spec.dataset_id"
    )
    op.execute("UPDATE dataset SET target_column = 'disease_cases' WHERE target_column IS NULL OR target_column = ''")
    op.alter_column("dataset", "target_column", nullable=False, server_default="disease_cases")


def downgrade() -> None:
    op.drop_column("dataset", "target_column")
