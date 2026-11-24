"""Record the principal that created each record (CLIM-1965).

Adds a nullable `created_by` to the records a token can create. Existing rows keep it empty,
and so do rows created while authentication is off. `create_db_and_tables` runs
`create_all` before Alembic, so on a fresh database the columns already exist and are skipped.

Revision ID: bfb92b05d718
Revises: 00e90ba1184f
Create Date: 2026-11-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from chap_core.database.migration_helpers import has_column

# revision identifiers, used by Alembic.
revision: str = "bfb92b05d718"
down_revision: Union[str, Sequence[str], None] = "00e90ba1184f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("prediction", "backtest", "dataset", "configuredmodeldb", "predictionsetup", "alert", "alertpolicy")


def upgrade() -> None:
    for table in TABLES:
        if not has_column(table, "created_by"):
            op.add_column(table, sa.Column("created_by", sa.String(), nullable=True))


def downgrade() -> None:
    for table in TABLES:
        if has_column(table, "created_by"):
            op.drop_column(table, "created_by")
