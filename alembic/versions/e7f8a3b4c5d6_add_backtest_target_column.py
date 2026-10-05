"""Include the dataset target column in backtest specification identity.

Revision ID: e7f8a3b4c5d6
Revises: d6e7f8a3b4c5
"""

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import has_column, has_unique_constraint

revision = "e7f8a3b4c5d6"
down_revision = "d6e7f8a3b4c5"
branch_labels = None
depends_on = None

TABLE = "backtestspecification"
CONSTRAINT = "uq_backtestspecification_params"
PARAMS = ["dataset_id", "n_periods", "n_splits", "stride", "n_retrain", "future_weather_provider"]


def upgrade() -> None:
    if not has_column(TABLE, "target_column"):
        op.add_column(TABLE, sa.Column("target_column", sa.String(), nullable=True))
    # The generic startup migration may have already added the column as an empty string.
    op.execute(
        "UPDATE backtestspecification SET target_column = 'disease_cases' "
        "WHERE target_column IS NULL OR target_column = ''"
    )
    op.alter_column(TABLE, "target_column", existing_type=sa.String(), nullable=False)
    if has_unique_constraint(TABLE, CONSTRAINT):
        op.drop_constraint(CONSTRAINT, TABLE, type_="unique")
    op.create_unique_constraint(CONSTRAINT, TABLE, [*PARAMS, "target_column"])


def downgrade() -> None:
    # Removing the target merges specifications that previously differed only by it.
    op.execute(
        "UPDATE backtest SET specification_id = grouped.id FROM "
        "(SELECT id, min(id) OVER (PARTITION BY " + ", ".join(PARAMS) + ") AS keep_id "
        "FROM backtestspecification) old JOIN backtestspecification grouped ON grouped.id = old.keep_id "
        "WHERE backtest.specification_id = old.id"
    )
    op.execute(
        "DELETE FROM backtestspecification WHERE id NOT IN "
        "(SELECT min(id) FROM backtestspecification GROUP BY " + ", ".join(PARAMS) + ")"
    )
    op.drop_constraint(CONSTRAINT, TABLE, type_="unique")
    op.drop_column(TABLE, "target_column")
    op.create_unique_constraint(CONSTRAINT, TABLE, PARAMS)
