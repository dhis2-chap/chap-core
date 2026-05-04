"""add_xai_tables

Adds:
- predictionexplanation table for storing per-(prediction, org_unit, period, method)
  XAI explanations produced by the surrogate / native-SHAP pipelines.
- provides_native_shap boolean column on modeltemplatedb so configured models can
  declare they emit shap_values.csv at predict time.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-04-27

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
    """Add provides_native_shap column and predictionexplanation table."""
    op.add_column(
        "modeltemplatedb",
        sa.Column(
            "provides_native_shap",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )

    op.create_table(
        "predictionexplanation",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("org_unit", sa.String(), nullable=False),
        sa.Column("period", sa.String(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column(
            "output_statistic",
            sa.String(),
            nullable=False,
            server_default="median",
        ),
        sa.Column("params", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("result", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default="completed",
        ),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column(
            "created",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["prediction.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_predictionexplanation_prediction_id",
        "predictionexplanation",
        ["prediction_id"],
    )
    op.create_index(
        "ix_predictionexplanation_prediction_id_method_org_unit",
        "predictionexplanation",
        ["prediction_id", "method", "org_unit"],
    )


def downgrade() -> None:
    """Drop predictionexplanation table and provides_native_shap column."""
    op.drop_index(
        "ix_predictionexplanation_prediction_id_method_org_unit",
        table_name="predictionexplanation",
    )
    op.drop_index(
        "ix_predictionexplanation_prediction_id",
        table_name="predictionexplanation",
    )
    op.drop_table("predictionexplanation")
    op.drop_column("modeltemplatedb", "provides_native_shap")
