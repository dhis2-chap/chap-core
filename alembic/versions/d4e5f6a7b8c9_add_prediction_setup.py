"""add_prediction_setup

Creates the predictionsetup table and adds a nullable prediction_setup_id
foreign key column on prediction. The legacy
configured_model_with_data_source_id column on prediction is left in place
for now; it will be migrated and dropped in a later revision.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictionsetup",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created", sa.DateTime(), nullable=True),
        sa.Column("backtest_id", sa.Integer(), nullable=False),
        sa.Column("configured_model_id", sa.Integer(), nullable=False),
        sa.Column("start_period", sa.String(), nullable=True),
        sa.Column("org_units", sa.JSON(), nullable=True),
        sa.Column("data_sources", sa.JSON(), nullable=True),
        sa.Column("period_type", sa.String(), nullable=True),
        sa.Column("schedule_cron_expression", sa.String(), nullable=True),
        sa.Column("schedule_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("data_import_mappings", sa.JSON(), nullable=True),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(["backtest_id"], ["backtest.id"]),
        sa.ForeignKeyConstraint(["configured_model_id"], ["configuredmodeldb.id"]),
    )

    op.add_column(
        "prediction",
        sa.Column("prediction_setup_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_prediction_prediction_setup",
        "prediction",
        "predictionsetup",
        ["prediction_setup_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_prediction_prediction_setup",
        "prediction",
        type_="foreignkey",
    )
    op.drop_column("prediction", "prediction_setup_id")
    op.drop_table("predictionsetup")
