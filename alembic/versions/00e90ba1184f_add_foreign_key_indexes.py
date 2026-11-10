"""add_foreign_key_indexes

Indexes the foreign keys of the three tables that hold one row per value: observation,
backtestforecast and predictionsamplesentry. Without them every dataset load, evaluation
entry and plot scans the whole table, so each read gets slower as the server stores more
datasets and backtests (CLIM-1791).

The column order follows the queries: observations are read for one dataset, often for one
feature and some org units; backtest forecasts for one backtest, often for some org units.

The indexes are also declared on the models. `create_db_and_tables` runs `create_all`
before Alembic, so on a fresh database they already exist and are skipped here.

Revision ID: 00e90ba1184f
Revises: b64628109792
Create Date: 2026-11-09

"""

from typing import Sequence, Union

from alembic import op

from chap_core.database.migration_helpers import has_index

# revision identifiers, used by Alembic.
revision: str = "00e90ba1184f"
down_revision: Union[str, Sequence[str], None] = "b64628109792"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEXES = [
    ("observation", "ix_observation_dataset_feature_org_unit", ["dataset_id", "feature_name", "org_unit"]),
    ("backtestforecast", "ix_backtestforecast_backtest_org_unit", ["backtest_id", "org_unit"]),
    ("predictionsamplesentry", "ix_predictionsamplesentry_prediction", ["prediction_id"]),
]


def upgrade() -> None:
    for table, name, columns in INDEXES:
        if not has_index(table, name):
            op.create_index(name, table, columns)


def downgrade() -> None:
    for table, name, _ in INDEXES:
        if has_index(table, name):
            op.drop_index(name, table_name=table)
