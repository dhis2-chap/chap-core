"""extract_backtest_specification

Move the evaluation parameters (n_periods, n_splits, stride, n_retrain,
future_weather_provider) off the backtest row and onto a deduplicated
``backtestspecification`` row that backtests point at, so that two backtests sharing a
specification are comparable by construction.

Existing rows are grouped by (dataset_id, and every one of those parameters) into one
specification each, and the specification's org_units is the union of the org_units of
the backtests that map to it. The provider is part of the key because two backtests
given different climate covariates for their forecast windows are not comparable, even
when every other parameter matches.

Two things about the migrated data are worth knowing, because neither can be
recovered here:

- The parameters of rows predating d0e1f2a3b4c5 were reconstructed from their
  forecasts rather than recorded. A split whose predictor returned nothing left no
  forecasts and is therefore invisible, so such a backtest was reconstructed with a
  lower n_splits and a wider stride. Two backtests that genuinely ran the same setup
  can carry different parameters and will land on different specifications. Legacy
  grouping is fragmented, and fragmented specifically for the models that misbehaved.
- backtest.org_units is accumulated from what the model produced, not from the
  filtered input, so for a model that dropped org units the copied set under-reports.
  The union across the backtests sharing a specification is the best estimate
  available: an org unit any of them forecast must have been in the input.
  Specifications created from now on resolve org_units from the filtered dataset.

Revision ID: f2a3b4c5d6e1
Revises: e1f2a3b4c5d6
Create Date: 2026-09-09

"""

import json
import logging
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f2a3b4c5d6e1"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")

# The parameters as of this revision, with the SQL type each column carries. A parameter
# added to BacktestParams later joins the specification's uniqueness key automatically,
# but still needs its own migration to move the column and rebuild the constraint; this
# list stays frozen at what was here.
PARAM_TYPES = {
    "n_periods": sa.Integer(),
    "n_splits": sa.Integer(),
    "stride": sa.Integer(),
    "n_retrain": sa.Integer(),
    "future_weather_provider": sa.String(),
}
PARAM_COLUMNS = tuple(PARAM_TYPES)
# BacktestParams defaults when this migration was written. Only reached on downgrade,
# for specifications that no backtest points at, so they stay local to this file.
DEFAULT_PARAMS = {"n_periods": 3, "n_splits": 7, "stride": 1, "n_retrain": 1, "future_weather_provider": "climatology"}
FOREIGN_KEY_NAME = "fk_backtest_specification_id_backtestspecification"


def _has_table(table: str) -> bool:
    return table in sa.inspect(op.get_bind()).get_table_names()


def _has_column(table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(op.get_bind()).get_columns(table))


def _foreign_key_name(table: str, referred_table: str) -> str | None:
    """Name of the constraint linking `table` to `referred_table`, whichever way it was created.

    create_all names the constraint by the Postgres default rather than by
    FOREIGN_KEY_NAME, so matching on the referred table is what makes this work on both
    a database that reached the new schema through create_all and one that did not.
    """
    for fk in sa.inspect(op.get_bind()).get_foreign_keys(table):
        if fk["referred_table"] == referred_table:
            return fk["name"]
    return None


def _org_units(value) -> list[str]:
    """Read a JSON org_units column, which comes back parsed or as text depending on the driver."""
    if value is None:
        return []
    return json.loads(value) if isinstance(value, str) else list(value)


def _backfill_specifications() -> None:
    """Create one specification per distinct (dataset, parameters) and point the backtests at it."""
    connection = op.get_bind()
    columns = ", ".join(PARAM_COLUMNS)
    rows = connection.execute(
        sa.text(f"SELECT id, dataset_id, org_units, {columns} FROM backtest ORDER BY id")
    ).fetchall()
    if not rows:
        return

    org_units_by_key: dict[tuple[int, ...], set[str]] = {}
    for row in rows:
        key = (row.dataset_id, *(getattr(row, column) for column in PARAM_COLUMNS))
        org_units_by_key.setdefault(key, set()).update(_org_units(row.org_units))

    logger.info(f"Extracting {len(org_units_by_key)} backtest specifications from {len(rows)} backtests")
    for key, org_units in org_units_by_key.items():
        params = dict(zip(PARAM_COLUMNS, key[1:], strict=True))
        specification_id = connection.execute(
            sa.text(
                f"INSERT INTO backtestspecification (dataset_id, {columns}, org_units) "
                f"VALUES (:dataset_id, :{', :'.join(PARAM_COLUMNS)}, :org_units) RETURNING id"
            ),
            {"dataset_id": key[0], **params, "org_units": json.dumps(sorted(org_units))},
        ).scalar_one()
        connection.execute(
            sa.text(
                "UPDATE backtest SET specification_id = :specification_id WHERE dataset_id = :dataset_id AND "
                + " AND ".join(f"{column} = :{column}" for column in PARAM_COLUMNS)
            ),
            {"specification_id": specification_id, "dataset_id": key[0], **params},
        )


def upgrade() -> None:
    """Create the specification table, move the parameters onto it, then drop them from backtest.

    Runs after e1f2a3b4c5d6, which put future_weather_provider on the backtest row; this
    revision moves it onto the specification with the rest of the parameters.

    Startup runs a generic metadata migration and create_all before Alembic, so the
    table and the specification_id column may already exist, the latter filled with 0
    rather than NULL. Both shapes end up the same way.
    """
    if not _has_table("backtestspecification"):
        op.create_table(
            "backtestspecification",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("dataset_id", sa.Integer(), nullable=False),
            *(sa.Column(column, type_, nullable=False) for column, type_ in PARAM_TYPES.items()),
            sa.Column("org_units", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(["dataset_id"], ["dataset.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(*(("dataset_id",) + PARAM_COLUMNS), name="uq_backtestspecification_params"),
        )
    if not _has_column("backtest", "specification_id"):
        op.add_column("backtest", sa.Column("specification_id", sa.Integer(), nullable=True))

    # The parameter columns are still there exactly when the backtests have not been
    # moved yet; create_all never drops columns, so it cannot have removed them. The
    # drop is guarded per column so that a database missing some of them, however it got
    # there, still ends up with none of them on backtest.
    if _has_column("backtest", "n_periods"):
        _backfill_specifications()
    for column in PARAM_COLUMNS:
        if _has_column("backtest", column):
            op.drop_column("backtest", column)

    op.alter_column("backtest", "specification_id", existing_type=sa.Integer(), nullable=False)
    if _foreign_key_name("backtest", "backtestspecification") is None:
        op.create_foreign_key(FOREIGN_KEY_NAME, "backtest", "backtestspecification", ["specification_id"], ["id"])


def downgrade() -> None:
    """Copy the parameters back onto backtest and drop the specification table.

    Lossy: a specification that no backtest points at has nowhere to go and is
    dropped with the table, and the org_units it resolved are lost.
    """
    for column, type_ in PARAM_TYPES.items():
        op.add_column("backtest", sa.Column(column, type_, nullable=True))
    op.execute(
        "UPDATE backtest SET "
        + ", ".join(f"{column} = s.{column}" for column in PARAM_COLUMNS)
        + " FROM backtestspecification s WHERE backtest.specification_id = s.id"
    )
    for column, default in DEFAULT_PARAMS.items():
        op.execute(
            sa.text(f"UPDATE backtest SET {column} = :default WHERE {column} IS NULL").bindparams(default=default)
        )
    for column, type_ in PARAM_TYPES.items():
        op.alter_column("backtest", column, existing_type=type_, nullable=False)
    foreign_key = _foreign_key_name("backtest", "backtestspecification")
    if foreign_key is not None:
        op.drop_constraint(foreign_key, "backtest", type_="foreignkey")
    op.drop_column("backtest", "specification_id")
    op.drop_table("backtestspecification")
