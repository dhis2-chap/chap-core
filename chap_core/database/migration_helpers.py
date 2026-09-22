"""Schema introspection for Alembic migrations.

Startup runs a generic metadata migration and ``create_all`` before Alembic, so a
migration cannot assume the objects it adds are absent. These helpers let a revision
check what already exists on the connection it runs against.
"""

import sqlalchemy as sa

from alembic import op


def has_table(table: str) -> bool:
    return table in sa.inspect(op.get_bind()).get_table_names()


def has_column(table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(op.get_bind()).get_columns(table))


def has_unique_constraint(table: str, name: str) -> bool:
    return any(item["name"] == name for item in sa.inspect(op.get_bind()).get_unique_constraints(table))


def foreign_key_name(table: str, referred_table: str) -> str | None:
    """Name of the constraint linking `table` to `referred_table`, whichever way it was created.

    create_all names a constraint by the Postgres default rather than by the name a
    migration chooses, so matching on the referred table works on both a database that
    reached the schema through create_all and one that did not.
    """
    for fk in sa.inspect(op.get_bind()).get_foreign_keys(table):
        if fk["referred_table"] == referred_table:
            return fk["name"]
    return None
