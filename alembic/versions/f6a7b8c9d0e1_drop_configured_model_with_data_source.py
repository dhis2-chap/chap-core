"""drop_configured_model_with_data_source

Removes the configured_model_with_data_source_id foreign key column on
prediction and drops the configuredmodelwithdatasource table. The
PredictionSetup table introduced in e5f6a7b8c9d0 replaces it.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, Sequence[str], None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "fk_prediction_configured_model_with_data_source",
        "prediction",
        type_="foreignkey",
    )
    op.drop_column("prediction", "configured_model_with_data_source_id")
    op.drop_table("configuredmodelwithdatasource")


def downgrade() -> None:
    op.create_table(
        "configuredmodelwithdatasource",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created", sa.DateTime(), nullable=True),
        sa.Column("configured_model_id", sa.Integer(), nullable=False),
        sa.Column("start_period", sa.String(), nullable=True),
        sa.Column("org_units", sa.JSON(), nullable=True),
        sa.Column("data_sources", sa.JSON(), nullable=True),
        sa.Column("period_type", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["configured_model_id"], ["configuredmodeldb.id"]),
    )
    op.add_column(
        "prediction",
        sa.Column("configured_model_with_data_source_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_prediction_configured_model_with_data_source",
        "prediction",
        "configuredmodelwithdatasource",
        ["configured_model_with_data_source_id"],
        ["id"],
    )
