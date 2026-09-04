"""version_templates_and_configurations

Template versions and configurations become immutable. A new version or a changed
configuration adds a row. It does not overwrite the row that a backtest points to.
The `is_live` flag shows the row that CHAP serves for a name.

`downgrade` makes each name unique again. It fails if a name has more than one row.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-08-18

"""

import hashlib
import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# Only for rows that CHAP made before the version became part of the identity.
LEGACY_UNVERSIONED_VERSION = "legacy-unversioned"

revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _backfill_configuration_digests() -> None:
    """Give each configuration the digest of the values it holds.

    Without a digest, the next seed replaces each configuration with an identical copy.
    The hash is local to this migration, so later code changes cannot change the result.
    """
    connection = op.get_bind()
    rows = connection.execute(
        sa.text("SELECT id, user_option_values, additional_continuous_covariates FROM configuredmodeldb")
    ).fetchall()
    for row in rows:
        payload = json.dumps(
            {
                "user_option_values": row.user_option_values or {},
                "additional_continuous_covariates": row.additional_continuous_covariates or [],
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        connection.execute(
            sa.text("UPDATE configuredmodeldb SET configuration_digest = :digest WHERE id = :id"),
            {"digest": digest, "id": row.id},
        )


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col["name"] == column for col in inspector.get_columns(table))


def _has_unique_constraint(table: str, name: str) -> bool:
    return any(item["name"] == name for item in sa.inspect(op.get_bind()).get_unique_constraints(table))


def upgrade() -> None:
    """Make (name, version) the template identity and add configuration digests.

    Startup runs a generic metadata migration before Alembic, so each column here
    may already exist — nullable, with generic defaults backfilled. Each column is
    therefore added only if missing, and the backfills and constraint swaps below
    give both paths the same outcome.
    """
    if not _has_column("modeltemplatedb", "source_digest"):
        # Nullable only for old rows. A new template must have a source digest.
        op.add_column("modeltemplatedb", sa.Column("source_digest", sa.String(), nullable=True))
    else:
        # The generic migration backfills text columns with ''. An unknown revision is NULL.
        op.execute(sa.text("UPDATE modeltemplatedb SET source_digest = NULL WHERE source_digest = ''"))

    if not _has_column("modeltemplatedb", "is_live"):
        op.add_column(
            "modeltemplatedb", sa.Column("is_live", sa.Boolean(), nullable=False, server_default=sa.true())
        )
        op.alter_column("modeltemplatedb", "is_live", server_default=None)
    else:
        # Names were unique until this migration, so each row is the live row for its name.
        op.execute(sa.text("UPDATE modeltemplatedb SET is_live = TRUE"))
        op.alter_column("modeltemplatedb", "is_live", existing_type=sa.Boolean(), nullable=False)

    # Version is part of the identity, so it cannot be null.
    op.execute(
        sa.text("UPDATE modeltemplatedb SET version = :version WHERE version IS NULL").bindparams(
            version=LEGACY_UNVERSIONED_VERSION
        )
    )
    op.alter_column("modeltemplatedb", "version", existing_type=sa.String(), nullable=False)
    op.execute("ALTER TABLE modeltemplatedb DROP CONSTRAINT IF EXISTS modeltemplatedb_name_key")
    if not _has_unique_constraint("modeltemplatedb", "uq_modeltemplatedb_name_version"):
        op.create_unique_constraint("uq_modeltemplatedb_name_version", "modeltemplatedb", ["name", "version"])

    if not _has_column("configuredmodeldb", "configuration_digest"):
        op.add_column(
            "configuredmodeldb", sa.Column("configuration_digest", sa.String(), nullable=False, server_default="")
        )

    if not _has_column("configuredmodeldb", "is_live"):
        op.add_column(
            "configuredmodeldb", sa.Column("is_live", sa.Boolean(), nullable=False, server_default=sa.true())
        )
        op.alter_column("configuredmodeldb", "is_live", server_default=None)
    else:
        op.execute(sa.text("UPDATE configuredmodeldb SET is_live = TRUE"))
        op.alter_column("configuredmodeldb", "is_live", existing_type=sa.Boolean(), nullable=False)

    _backfill_configuration_digests()
    op.alter_column(
        "configuredmodeldb", "configuration_digest", existing_type=sa.String(), nullable=False, server_default=None
    )
    op.execute("ALTER TABLE configuredmodeldb DROP CONSTRAINT IF EXISTS configuredmodeldb_name_key")
    if not _has_unique_constraint("configuredmodeldb", "uq_configuredmodeldb_template_name_digest"):
        op.create_unique_constraint(
            "uq_configuredmodeldb_template_name_digest",
            "configuredmodeldb",
            ["model_template_id", "name", "configuration_digest"],
        )


def downgrade() -> None:
    """Restore globally unique names and drop the versioning columns."""
    op.drop_constraint("uq_configuredmodeldb_template_name_digest", "configuredmodeldb", type_="unique")
    op.create_unique_constraint("configuredmodeldb_name_key", "configuredmodeldb", ["name"])
    op.drop_column("configuredmodeldb", "is_live")
    op.drop_column("configuredmodeldb", "configuration_digest")

    op.drop_constraint("uq_modeltemplatedb_name_version", "modeltemplatedb", type_="unique")
    op.create_unique_constraint("modeltemplatedb_name_key", "modeltemplatedb", ["name"])
    op.alter_column("modeltemplatedb", "version", existing_type=sa.String(), nullable=True)
    op.drop_column("modeltemplatedb", "is_live")
    op.drop_column("modeltemplatedb", "source_digest")
