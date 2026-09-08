"""add_backtest_params

Store the evaluation parameters a backtest ran with (n_periods, n_splits, stride,
n_retrain) on the backtest row. Legacy rows are reconstructed from their forecasts:
one split per distinct last_seen_period, n_periods from the periods forecast per
split, stride from the smallest gap between splits. n_retrain is 1 for every legacy
row because the REST path never forwarded it.

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-09-08

"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa

from alembic import op

revision: str = "d0e1f2a3b4c5"
down_revision: str | Sequence[str] | None = "c9d0e1f2a3b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")

PARAM_COLUMNS = ("n_periods", "n_splits", "stride", "n_retrain")
# BacktestParams defaults when this migration was written. Only used for rows whose
# forecasts do not allow the value to be derived, so they stay local to this file.
DEFAULT_PARAMS = {"n_periods": 3, "n_splits": 7, "stride": 1, "n_retrain": 1}
DEFAULT_WEEKLY_STRIDE = 4


def _period_index(period_id: str) -> int | None:
    """Ordinal of a period id in units of its own kind: months for YYYYMM, weeks for YYYYWnn / YYYYSunWnn."""
    if "W" in period_id:
        year, week = period_id.split("W")
        return datetime.strptime(f"{year[:4]}-W{int(week):02d}-1", "%G-W%V-%u").toordinal() // 7
    if len(period_id) == 6 and period_id.isdigit():
        return int(period_id[:4]) * 12 + int(period_id[4:]) - 1
    return None


def derive_stride(split_ids: list[str]) -> int:
    """Smallest gap between consecutive splits. A split that produced no forecasts leaves a wider gap."""
    indices = [_period_index(split_id) for split_id in sorted(set(split_ids))]
    if len(indices) < 2 or None in indices:
        return DEFAULT_WEEKLY_STRIDE if split_ids and "W" in split_ids[0] else DEFAULT_PARAMS["stride"]
    return min(b - a for a, b in zip(indices[:-1], indices[1:], strict=True))  # type: ignore[operator]


def _derive_params(splits: list[tuple[str, int]]) -> dict[str, int]:
    """Parameters for one backtest from its (last_seen_period, periods forecast) per split."""
    if not splits:
        return dict(DEFAULT_PARAMS)
    split_ids = [split_id for split_id, _ in splits]
    return {
        "n_periods": max(n_periods for _, n_periods in splits),
        "n_splits": len(split_ids),
        "stride": derive_stride(split_ids),
        "n_retrain": 1,
    }


def _backfill_backtest_params() -> None:
    connection = op.get_bind()
    missing = " OR ".join(f"{column} IS NULL OR {column} = 0" for column in PARAM_COLUMNS)
    ids = [row.id for row in connection.execute(sa.text(f"SELECT id FROM backtest WHERE {missing}")).fetchall()]
    if not ids:
        return
    splits: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for row in connection.execute(
        sa.text(
            "SELECT backtest_id, last_seen_period, COUNT(DISTINCT period) AS n_periods "
            "FROM backtestforecast GROUP BY backtest_id, last_seen_period"
        )
    ).fetchall():
        splits[row.backtest_id].append((row.last_seen_period, row.n_periods))
    without_forecasts = [backtest_id for backtest_id in ids if backtest_id not in splits]
    if without_forecasts:
        logger.warning(f"Backtests {without_forecasts} have no forecasts; storing default parameters")
    for backtest_id in ids:
        params = _derive_params(splits.get(backtest_id, []))
        connection.execute(
            sa.text(
                "UPDATE backtest SET n_periods = :n_periods, n_splits = :n_splits, "
                "stride = :stride, n_retrain = :n_retrain WHERE id = :id"
            ),
            {**params, "id": backtest_id},
        )


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col["name"] == column for col in inspector.get_columns(table))


def upgrade() -> None:
    """Add the parameter columns, reconstruct them for existing rows, then make them NOT NULL.

    Startup runs a generic metadata migration before Alembic, so each column may
    already exist with 0 in every row. Both shapes are backfilled the same way.
    """
    for column in PARAM_COLUMNS:
        if not _has_column("backtest", column):
            op.add_column("backtest", sa.Column(column, sa.Integer(), nullable=True))
    _backfill_backtest_params()
    for column in PARAM_COLUMNS:
        op.alter_column("backtest", column, existing_type=sa.Integer(), nullable=False)


def downgrade() -> None:
    """Drop the parameter columns."""
    for column in reversed(PARAM_COLUMNS):
        op.drop_column("backtest", column)
