"""The introspection helpers migrations use to tell what create_all already put in place."""

import pytest
import sqlalchemy as sa
from alembic import op
from alembic.migration import MigrationContext
from alembic.operations import Operations

from chap_core.database.migration_helpers import foreign_key_name, has_column, has_table, has_unique_constraint


@pytest.fixture
def operations(clean_engine):
    """Bind `alembic.op` to the fully created schema, as it is inside a running migration."""
    with clean_engine.connect() as connection, Operations.context(MigrationContext.configure(connection)):
        yield


def test_has_table(operations):
    assert has_table("backtest")
    assert not has_table("no_such_table")


def test_has_column(operations):
    assert has_column("backtest", "chap_version")
    assert not has_column("backtest", "no_such_column")


def test_has_unique_constraint(operations):
    assert has_unique_constraint("backtestspecification", "uq_backtestspecification_params")
    assert not has_unique_constraint("backtestspecification", "no_such_constraint")


def test_foreign_key_name(operations):
    # sqlite leaves the keys create_all makes unnamed, so the positive case needs its own
    # table. sqlite commits before DDL, so it is dropped explicitly rather than rolled back.
    op.create_table(
        "scratch",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("backtest_id", sa.Integer()),
        sa.ForeignKeyConstraint(["backtest_id"], ["backtest.id"], name="fk_scratch_backtest"),
    )
    try:
        assert foreign_key_name("scratch", "backtest") == "fk_scratch_backtest"
        assert foreign_key_name("scratch", "dataset") is None
    finally:
        op.drop_table("scratch")
