"""add_prediction_setup_observation

Creates the predictionsetupobservation table: the observed disease cases each run of a
PredictionSetup uploads, one row per (setup, org unit, period), so the setup's
predictions can be scored against them.

The foreign key is CASCADE: the values only exist to monitor their setup.

Revision ID: e7f8a3b4c5d6
Revises: d6e7f8a3b4c5
Create Date: 2026-10-02

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from chap_core.database.migration_helpers import has_table

# revision identifiers, used by Alembic.
revision: str = "e7f8a3b4c5d6"
down_revision: Union[str, Sequence[str], None] = "d6e7f8a3b4c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not has_table("predictionsetupobservation"):
        op.create_table(
            "predictionsetupobservation",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("prediction_setup_id", sa.Integer(), nullable=False),
            sa.Column("org_unit", sa.String(), nullable=False),
            sa.Column("period", sa.String(), nullable=False),
            sa.Column("disease_cases", sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(
                ["prediction_setup_id"],
                ["predictionsetup.id"],
                ondelete="CASCADE",
                name="fk_predictionsetupobservation_prediction_setup",
            ),
            sa.UniqueConstraint(
                "prediction_setup_id",
                "org_unit",
                "period",
                name="uq_predictionsetupobservation_setup_org_unit_period",
            ),
        )


def downgrade() -> None:
    op.drop_table("predictionsetupobservation")
