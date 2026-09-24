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

from chap_core.database.migration_helpers import foreign_key_name, has_column

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not has_column("prediction", "configured_model_with_data_source_id"):
        op.add_column(
            "prediction",
            sa.Column("configured_model_with_data_source_id", sa.Integer(), nullable=True),
        )
    if foreign_key_name("prediction", "configuredmodelwithdatasource") is None:
        # The generic startup migration fills the column with 0 on existing rows, which
        # no configured model with data source has, so clear dangling ids first.
        op.execute(
            sa.text(
                "UPDATE prediction SET configured_model_with_data_source_id = NULL "
                "WHERE configured_model_with_data_source_id NOT IN (SELECT id FROM configuredmodelwithdatasource)"
            )
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
