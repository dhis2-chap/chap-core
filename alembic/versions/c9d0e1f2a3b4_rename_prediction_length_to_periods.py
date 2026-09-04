"""rename_prediction_length_to_periods

Aligns the model-template horizon columns with chapkit, which calls them
`min_prediction_periods` / `max_prediction_periods`.

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-09-01

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c9d0e1f2a3b4"
down_revision: str | Sequence[str] | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "modeltemplatedb"
_RENAMES = (
    ("min_prediction_length", "min_prediction_periods"),
    ("max_prediction_length", "max_prediction_periods"),
)


def _has_column(column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(op.get_bind()).get_columns(TABLE))


def _move(source: str, target: str) -> None:
    """Move a horizon column's values from `source` to `target`.

    Startup runs a generic metadata migration before Alembic. That migration adds
    any column the models declare but the database lacks, so by the time this runs,
    `target` may already exist as an empty column beside a populated `source`.
    Renaming would fail there and dropping would lose the values, so that case is
    handled by copying across before the old column goes away.
    """
    if not _has_column(source):
        return
    if not _has_column(target):
        op.alter_column(TABLE, source, new_column_name=target, existing_type=sa.Integer(), existing_nullable=True)
        return
    op.execute(sa.text(f"UPDATE {TABLE} SET {target} = {source} WHERE {target} IS NULL"))
    op.drop_column(TABLE, source)


def upgrade() -> None:
    for old, new in _RENAMES:
        _move(old, new)


def downgrade() -> None:
    for old, new in _RENAMES:
        _move(new, old)
