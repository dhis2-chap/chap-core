"""add_prediction_configured_model_with_data_source_fk

Adds nullable configured_model_with_data_source_id foreign key column to
the prediction table.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-04-21

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


def downgrade() -> None:
    op.drop_constraint(
        "fk_prediction_configured_model_with_data_source",
        "prediction",
        type_="foreignkey",
    )
    op.drop_column("prediction", "configured_model_with_data_source_id")
