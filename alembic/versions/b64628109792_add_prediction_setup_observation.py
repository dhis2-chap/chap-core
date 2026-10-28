"""add_prediction_setup_observation

Creates the predictionsetupobservation table: the observed disease cases each run of a
PredictionSetup uploads, one row per (setup, org unit, period), so the setup's
predictions can be scored against them. Adds prediction.baseline_for_prediction_id, which
links the baseline model's forecast made with each setup prediction to that prediction,
so monitoring can compare the two.

The foreign keys are CASCADE: the values and baseline forecasts only exist to monitor
their setup.

Revision ID: b64628109792
Revises: e6f7a8b9c0d1
Create Date: 2026-10-02

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from chap_core.database.migration_helpers import foreign_key_name, has_column, has_table

# revision identifiers, used by Alembic.
revision: str = "b64628109792"
down_revision: Union[str, Sequence[str], None] = "e6f7a8b9c0d1"
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
    if not has_column("prediction", "baseline_for_prediction_id"):
        op.add_column("prediction", sa.Column("baseline_for_prediction_id", sa.Integer(), nullable=True))
    if foreign_key_name("prediction", "prediction") is None:
        # The generic startup migration fills the column with 0 on existing rows, which no prediction has.
        op.execute(sa.text("UPDATE prediction SET baseline_for_prediction_id = NULL"))
        op.create_foreign_key(
            "fk_prediction_baseline_for_prediction",
            "prediction",
            "prediction",
            ["baseline_for_prediction_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    foreign_key = foreign_key_name("prediction", "prediction")
    if foreign_key is not None:
        op.drop_constraint(foreign_key, "prediction", type_="foreignkey")
    op.drop_column("prediction", "baseline_for_prediction_id")
    op.drop_table("predictionsetupobservation")
