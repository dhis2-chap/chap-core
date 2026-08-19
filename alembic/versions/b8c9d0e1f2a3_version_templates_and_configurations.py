"""version_templates_and_configurations

Makes template versions and configurations immutable. Uniqueness moves from
`name` to `(name, version)` on modeltemplatedb and to
`(model_template_id, name, configuration_digest)` on configuredmodeldb, so a new
version or an edited configuration becomes a new row instead of rewriting the row
that finished backtests still point at. Both tables get an `is_live` flag marking
the row currently served under a given name.

Note that `downgrade` restores globally unique names, so it only succeeds while
each name still has a single row. Superseded rows must be resolved by hand first.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-08-18

"""

import hashlib
import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# Existing rows predate versioned template identity. This is deliberately local
# to the migration: new registrations must supply an actual version.
LEGACY_UNVERSIONED_VERSION = "legacy-unversioned"

revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _backfill_configuration_digests() -> None:
    """Give existing configurations the digest of the values they already hold.

    Without this, every configured model would look edited on the next seeding and
    be superseded by an identical copy of itself.

    The hashing is inlined rather than imported from
    ``chap_core.database.model_templates_and_config_tables.compute_configuration_digest``
    so that a later change to that function cannot retroactively alter what this
    migration writes.
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


def upgrade() -> None:
    """Make (name, version) the template identity and add configuration digests."""
    # This remains nullable only for rows created before source provenance was
    # enforced. New template inserts are rejected by SessionWrapper unless they
    # supply an immutable source digest.
    op.add_column("modeltemplatedb", sa.Column("source_digest", sa.String(), nullable=True))
    # every pre-migration row is the only one under its name, so all of them are live
    op.add_column(
        "modeltemplatedb", sa.Column("is_live", sa.Boolean(), nullable=False, server_default=sa.true())
    )
    op.alter_column("modeltemplatedb", "is_live", server_default=None)
    # version is part of the template identity, so it can never be null
    op.execute(
        sa.text("UPDATE modeltemplatedb SET version = :version WHERE version IS NULL").bindparams(
            version=LEGACY_UNVERSIONED_VERSION
        )
    )
    op.alter_column("modeltemplatedb", "version", existing_type=sa.String(), nullable=False)
    op.execute("ALTER TABLE modeltemplatedb DROP CONSTRAINT IF EXISTS modeltemplatedb_name_key")
    op.create_unique_constraint("uq_modeltemplatedb_name_version", "modeltemplatedb", ["name", "version"])

    op.add_column(
        "configuredmodeldb", sa.Column("configuration_digest", sa.String(), nullable=False, server_default="")
    )
    op.add_column(
        "configuredmodeldb", sa.Column("is_live", sa.Boolean(), nullable=False, server_default=sa.true())
    )
    op.alter_column("configuredmodeldb", "is_live", server_default=None)
    _backfill_configuration_digests()
    op.alter_column("configuredmodeldb", "configuration_digest", server_default=None)
    op.execute("ALTER TABLE configuredmodeldb DROP CONSTRAINT IF EXISTS configuredmodeldb_name_key")
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
