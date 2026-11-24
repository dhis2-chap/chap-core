"""Add the hporundb table and link configured models to the HPO run that produced them.

Startup runs create_all before Alembic, so the table may already exist. The
hpo_run_id column is added here only, since startup adds no columns past the
generic schema revision.

Revision ID: a1c2e3f4b5d6
Revises: bfb92b05d718
"""

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import foreign_key_name, has_column, has_table

revision = "a1c2e3f4b5d6"
down_revision = "bfb92b05d718"
branch_labels = None
depends_on = None

FOREIGN_KEY_NAME = "fk_configuredmodeldb_hpo_run_id_hporundb"


def upgrade() -> None:
    if not has_table("hporundb"):
        op.create_table(
            "hporundb",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("searcher", sa.String(), nullable=False),
            sa.Column("metric", sa.String(), nullable=False),
            sa.Column("direction", sa.String(), nullable=False),
            sa.Column("search_space", sa.JSON(), nullable=True),
            sa.Column("max_trials", sa.Integer(), nullable=True),
            sa.Column("seed", sa.Integer(), nullable=True),
            sa.Column("best_params", sa.JSON(), nullable=True),
            sa.Column("best_score", sa.Float(), nullable=False),
            sa.Column("n_trials", sa.Integer(), nullable=False),
            sa.Column("n_successful_trials", sa.Integer(), nullable=False),
            sa.Column("n_failed_trials", sa.Integer(), nullable=False),
            sa.Column("elapsed_seconds", sa.Float(), nullable=False),
            sa.Column("stop_reason", sa.String(), nullable=False),
            sa.Column("leaderboard", sa.JSON(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
    if not has_column("configuredmodeldb", "hpo_run_id"):
        op.add_column("configuredmodeldb", sa.Column("hpo_run_id", sa.Integer(), nullable=True))
    if foreign_key_name("configuredmodeldb", "hporundb") is None:
        op.create_foreign_key(
            FOREIGN_KEY_NAME, "configuredmodeldb", "hporundb", ["hpo_run_id"], ["id"], ondelete="SET NULL"
        )


def downgrade() -> None:
    foreign_key = foreign_key_name("configuredmodeldb", "hporundb")
    if foreign_key is not None:
        op.drop_constraint(foreign_key, "configuredmodeldb", type_="foreignkey")
    if has_column("configuredmodeldb", "hpo_run_id"):
        op.drop_column("configuredmodeldb", "hpo_run_id")
    if has_table("hporundb"):
        op.drop_table("hporundb")
