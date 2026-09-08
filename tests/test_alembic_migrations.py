"""
Tests for alembic migration chain (CLIM-349).

Spins up a real PostgreSQL container via testcontainers, creates the baseline
schema (as it existed before alembic), then runs the full upgrade/downgrade cycle.

Requires Docker to be available. Skips automatically if Docker is not running.

New migrations are automatically tested since upgrade always targets "head".
"""

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlmodel import SQLModel

# Import all models so SQLModel.metadata is fully populated
from chap_core.database.dataset_tables import DataSet, Observation  # noqa: F401
from chap_core.database.feature_tables import FeatureSource, FeatureType  # noqa: F401
from chap_core.database.model_spec_tables import ModelFeatureLink, ModelSpec  # noqa: F401
from chap_core.database.model_templates_and_config_tables import (  # noqa: F401
    ConfiguredModelDB,
    ModelTemplateDB,
)
from chap_core.database.tables import (  # noqa: F401
    Backtest,
    BacktestForecast,
    BacktestMetric,
    Prediction,
    PredictionSamplesEntry,
    PredictionSetup,
)

PROJECT_ROOT = Path(__file__).parent.parent
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

# Columns added by alembic migrations (not in the baseline schema).
# When simulating a pre-alembic database we create all tables from
# SQLModel metadata then drop these columns so the migration can re-add them.
_COLUMNS_ADDED_BY_MIGRATIONS = [
    ("modeltemplatedb", "archived"),
    ("modeltemplatedb", "source_digest"),
    ("modeltemplatedb", "is_live"),
    ("configuredmodeldb", "configuration_digest"),
    ("configuredmodeldb", "is_live"),
    ("prediction", "prediction_setup_id"),
    ("backtest", "max_horizon_distance"),
]

# Tables added by alembic migrations (not in the baseline schema).
# These are dropped after create_all so the migration can re-create them.
_TABLES_ADDED_BY_MIGRATIONS = [
    "predictionsetup",
]

# Unique constraints from migrations, with the baseline constraint that each one replaced.
# create_all makes the new shape, so the baseline must go back to the old shape.
_CONSTRAINTS_REPLACED_BY_MIGRATIONS = [
    ("modeltemplatedb", "uq_modeltemplatedb_name_version", "modeltemplatedb_name_key", "name"),
    ("configuredmodeldb", "uq_configuredmodeldb_template_name_digest", "configuredmodeldb_name_key", "name"),
]

# Columns that a migration made NOT NULL. This makes the test run the backfill.
_COLUMNS_MADE_NOT_NULL_BY_MIGRATIONS = [
    ("modeltemplatedb", "version"),
]

# Columns a migration renamed, as (table, baseline name, current name). create_all
# makes the current name, so the baseline has to be put back to the old one for the
# rename to be exercised.
_COLUMNS_RENAMED_BY_MIGRATIONS = [
    ("modeltemplatedb", "min_prediction_length", "min_prediction_periods"),
    ("modeltemplatedb", "max_prediction_length", "max_prediction_periods"),
]


def _pg_container():
    """Create and start a PostgreSQL testcontainer."""
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers[postgres] not installed")

    try:
        container = PostgresContainer("postgres:17-alpine")
        container.start()
        return container
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture(scope="module")
def pg():
    """Module-scoped PostgreSQL container fixture."""
    container = _pg_container()
    yield container
    container.stop()


@pytest.fixture(scope="module")
def engine(pg):
    """SQLAlchemy engine connected to the test PostgreSQL database."""
    eng = sa.create_engine(pg.get_connection_url())
    yield eng
    eng.dispose()


def _make_alembic_cfg(engine):
    """Create an Alembic config that passes the engine connection directly.

    This bypasses env.py's CHAP_DATABASE_URL override by providing the
    connection via config.attributes, which env.py checks first.
    """
    from alembic.config import Config

    cfg = Config(str(ALEMBIC_INI))
    cfg.attributes["connection"] = engine
    return cfg


def _create_baseline_schema(engine):
    """Create the database schema as it existed at the alembic baseline.

    Creates all tables from SQLModel metadata, then drops columns that
    were added by subsequent alembic migrations. This simulates the state
    of a database before alembic migrations were applied.
    """
    SQLModel.metadata.create_all(engine)

    with engine.connect() as conn:
        # Drop columns before tables so FKs pointing at soon-to-be-dropped
        # tables are removed first.
        for table, column in _COLUMNS_ADDED_BY_MIGRATIONS:
            conn.execute(sa.text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}"))
        for table in _TABLES_ADDED_BY_MIGRATIONS:
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {table}"))
        for table, new_constraint, baseline_constraint, baseline_columns in _CONSTRAINTS_REPLACED_BY_MIGRATIONS:
            conn.execute(sa.text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {new_constraint}"))
            conn.execute(
                sa.text(f"ALTER TABLE {table} ADD CONSTRAINT {baseline_constraint} UNIQUE ({baseline_columns})")
            )
        for table, column in _COLUMNS_MADE_NOT_NULL_BY_MIGRATIONS:
            conn.execute(sa.text(f"ALTER TABLE {table} ALTER COLUMN {column} DROP NOT NULL"))
        for table, baseline_name, current_name in _COLUMNS_RENAMED_BY_MIGRATIONS:
            conn.execute(sa.text(f"ALTER TABLE {table} RENAME COLUMN {current_name} TO {baseline_name}"))
        conn.commit()


@pytest.mark.slow
class TestAlembicMigrations:
    """Test the full alembic migration chain against a real PostgreSQL database."""

    def test_upgrade_to_head(self, engine):
        """
        Simulate a pre-alembic database, stamp baseline, then upgrade to head.
        """
        from alembic import command
        from alembic.script import ScriptDirectory

        alembic_cfg = _make_alembic_cfg(engine)

        # Create baseline schema (tables without columns added by migrations)
        _create_baseline_schema(engine)

        # Stamp the baseline revision so alembic knows where we are
        command.stamp(alembic_cfg, "fe59a33965ed")

        # Upgrade to head (applies all migrations after baseline)
        command.upgrade(alembic_cfg, "head")

        # Verify we are at head
        script = ScriptDirectory.from_config(alembic_cfg)
        head_rev = script.get_current_head()

        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT version_num FROM alembic_version"))
            current = result.scalar_one()
            assert current == head_rev, f"Expected head {head_rev}, got {current}"

    def test_downgrade_to_base_and_upgrade_again(self, engine):
        """
        After upgrading, downgrade back to baseline then upgrade again.
        Verifies downgrade() functions work correctly.
        """
        from alembic import command
        from alembic.script import ScriptDirectory

        alembic_cfg = _make_alembic_cfg(engine)

        # Downgrade to baseline
        command.downgrade(alembic_cfg, "fe59a33965ed")

        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT version_num FROM alembic_version"))
            current = result.scalar_one()
            assert current == "fe59a33965ed"

        # Upgrade back to head
        command.upgrade(alembic_cfg, "head")

        script = ScriptDirectory.from_config(alembic_cfg)
        head_rev = script.get_current_head()

        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT version_num FROM alembic_version"))
            current = result.scalar_one()
            assert current == head_rev

    def test_upgrade_applies_after_generic_startup_migration(self, engine):
        """
        create_db_and_tables adds missing columns from model metadata before it
        runs Alembic, so the columns of the versioning migration can already
        exist. The migration must still run its backfills and constraint swap.
        """
        from alembic import command

        alembic_cfg = _make_alembic_cfg(engine)

        with engine.connect() as conn:
            conn.execute(sa.text("DROP SCHEMA public CASCADE"))
            conn.execute(sa.text("CREATE SCHEMA public"))
            conn.commit()
        _create_baseline_schema(engine)
        command.stamp(alembic_cfg, "fe59a33965ed")
        # The state of a deployment on the previous release.
        command.upgrade(alembic_cfg, "a7b8c9d0e1f2")

        with engine.connect() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO modeltemplatedb "
                    "(name, display_name, description, author_note, author_assessed_status, author, "
                    "supported_period_type, target, allow_free_additional_continuous_covariates, requires_geo, "
                    "uses_chapkit) "
                    "VALUES ('legacy_model', 'Legacy', 'legacy', 'note', 'gray', 'author', "
                    "'any', 'disease_cases', false, false, false)"
                )
            )
            # What the generic startup migration does before Alembic runs.
            for statement in [
                "ALTER TABLE modeltemplatedb ADD COLUMN source_digest VARCHAR",
                "UPDATE modeltemplatedb SET source_digest = ''",
                "ALTER TABLE modeltemplatedb ADD COLUMN is_live BOOLEAN",
                "UPDATE modeltemplatedb SET is_live = true",
                "ALTER TABLE configuredmodeldb ADD COLUMN configuration_digest VARCHAR",
                "UPDATE configuredmodeldb SET configuration_digest = ''",
                "ALTER TABLE configuredmodeldb ADD COLUMN is_live BOOLEAN",
                "UPDATE configuredmodeldb SET is_live = true",
            ]:
                conn.execute(sa.text(statement))
            conn.commit()

        command.upgrade(alembic_cfg, "head")

        with engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT version, source_digest, is_live FROM modeltemplatedb WHERE name = 'legacy_model'")
            ).one()
            assert row.version == "legacy-unversioned"
            assert row.source_digest is None
            assert row.is_live is True

        constraints = {c["name"] for c in sa.inspect(engine).get_unique_constraints("modeltemplatedb")}
        assert "uq_modeltemplatedb_name_version" in constraints
        assert "modeltemplatedb_name_key" not in constraints

    @pytest.mark.parametrize("generic_migration_ran_first", [False, True])
    def test_prediction_horizon_values_survive_rename(self, engine, generic_migration_ran_first):
        """The horizons a template declares must still be there under the new names.

        Startup adds missing columns from model metadata before Alembic runs, so the
        renamed column can already exist, empty, beside the populated old one. A plain
        rename would fail there, and dropping the old column would lose the horizons,
        which decide whether a model can serve a requested backtest length.
        """
        from alembic import command

        alembic_cfg = _make_alembic_cfg(engine)

        with engine.connect() as conn:
            conn.execute(sa.text("DROP SCHEMA public CASCADE"))
            conn.execute(sa.text("CREATE SCHEMA public"))
            conn.commit()
        _create_baseline_schema(engine)
        command.stamp(alembic_cfg, "fe59a33965ed")
        command.upgrade(alembic_cfg, "b8c9d0e1f2a3")

        with engine.connect() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO modeltemplatedb "
                    "(name, version, display_name, description, author_note, author_assessed_status, author, "
                    "supported_period_type, target, allow_free_additional_continuous_covariates, requires_geo, "
                    "uses_chapkit, is_live, archived, min_prediction_length, max_prediction_length) "
                    "VALUES ('bounded_model', 'v1', 'Bounded', 'desc', 'note', 'gray', 'author', "
                    "'any', 'disease_cases', false, false, false, true, false, 2, 6)"
                )
            )
            if generic_migration_ran_first:
                for column in ("min_prediction_periods", "max_prediction_periods"):
                    conn.execute(sa.text(f"ALTER TABLE modeltemplatedb ADD COLUMN {column} INTEGER"))
            conn.commit()

        command.upgrade(alembic_cfg, "head")

        columns = {col["name"] for col in sa.inspect(engine).get_columns("modeltemplatedb")}
        assert "min_prediction_length" not in columns
        assert "max_prediction_length" not in columns

        with engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT min_prediction_periods, max_prediction_periods "
                    "FROM modeltemplatedb WHERE name = 'bounded_model'"
                )
            ).one()
        assert row.min_prediction_periods == 2
        assert row.max_prediction_periods == 6

    def test_unversioned_create_all_schema_is_bootstrapped_to_head(self, engine):
        """A legacy create_all database must still run the versioning migration."""
        from alembic.script import ScriptDirectory

        from chap_core.database.database import _run_alembic_migrations

        with engine.connect() as conn:
            conn.execute(sa.text("DROP SCHEMA public CASCADE"))
            conn.execute(sa.text("CREATE SCHEMA public"))
            conn.commit()

        # Startup's generic migration and create_all call give an unversioned
        # database current columns and tables, but cannot replace constraints on
        # existing tables. Startup must still run the versioning migration.
        SQLModel.metadata.create_all(engine)
        with engine.connect() as conn:
            for table, new_constraint, baseline_constraint, baseline_columns in _CONSTRAINTS_REPLACED_BY_MIGRATIONS:
                conn.execute(sa.text(f"ALTER TABLE {table} DROP CONSTRAINT {new_constraint}"))
                conn.execute(
                    sa.text(f"ALTER TABLE {table} ADD CONSTRAINT {baseline_constraint} UNIQUE ({baseline_columns})")
                )
            conn.commit()

        _run_alembic_migrations(engine)

        script = ScriptDirectory.from_config(_make_alembic_cfg(engine))
        with engine.connect() as conn:
            current = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()
        assert current == script.get_current_head()

        template_constraints = {
            constraint["name"] for constraint in sa.inspect(engine).get_unique_constraints("modeltemplatedb")
        }
        assert "uq_modeltemplatedb_name_version" in template_constraints
        assert "modeltemplatedb_name_key" not in template_constraints

        configured_model_constraints = {
            constraint["name"] for constraint in sa.inspect(engine).get_unique_constraints("configuredmodeldb")
        }
        assert "uq_configuredmodeldb_template_name_digest" in configured_model_constraints
        assert "configuredmodeldb_name_key" not in configured_model_constraints

    def test_all_revisions_have_downgrade(self):
        """Verify every migration revision defines a non-empty downgrade."""
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        cfg = Config(str(ALEMBIC_INI))
        script = ScriptDirectory.from_config(cfg)

        for rev in script.walk_revisions():
            module = rev.module
            downgrade_fn = getattr(module, "downgrade", None)
            assert downgrade_fn is not None, f"Revision {rev.revision} missing downgrade()"

    def test_migration_history_is_linear(self):
        """Verify no branch points exist in the migration chain."""
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        cfg = Config(str(ALEMBIC_INI))
        script = ScriptDirectory.from_config(cfg)
        branches = list(script.get_bases())
        assert len(branches) == 1, f"Expected 1 base, found {len(branches)}: {branches}"

        heads = list(script.get_heads())
        assert len(heads) == 1, f"Expected 1 head, found {len(heads)}: {heads}"
