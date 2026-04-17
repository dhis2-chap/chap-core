"""add_configured_model_with_data_source

Adds configuredmodelwithdatasource table for storing configured models
with their associated data source metadata.

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-17

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6g7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "configuredmodelwithdatasource",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created", sa.DateTime(), nullable=True),
        sa.Column("configured_model_id", sa.Integer(), nullable=False),
        sa.Column("start_period", sa.String(), nullable=True),
        sa.Column("org_units", sa.JSON(), nullable=True),
        sa.Column("data_source_mapping", sa.JSON(), nullable=True),
        sa.Column("period_type", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["configured_model_id"], ["configuredmodeldb.id"]),
    )


def downgrade() -> None:
    op.drop_table("configuredmodelwithdatasource")
