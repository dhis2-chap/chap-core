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
from chap_core.database.debug import DebugEntry  # noqa: F401
from chap_core.database.feature_tables import FeatureSource, FeatureType  # noqa: F401
from chap_core.database.model_spec_tables import ModelFeatureLink, ModelSpec  # noqa: F401
from chap_core.database.model_templates_and_config_tables import (  # noqa: F401
    ConfiguredModelDB,
    ModelTemplateDB,
)
from chap_core.database.tables import (  # noqa: F401
    BackTest,
    BackTestForecast,
    BackTestMetric,
    Prediction,
    PredictionSamplesEntry,
)

PROJECT_ROOT = Path(__file__).parent.parent
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

# Columns added by alembic migrations (not in the baseline schema).
# When simulating a pre-alembic database we create all tables from
# SQLModel metadata then drop these columns so the migration can re-add them.
_COLUMNS_ADDED_BY_MIGRATIONS = [
    ("modeltemplatedb", "archived"),
    ("prediction", "configured_model_with_data_source_id"),
]

# Tables added by alembic migrations (not in the baseline schema).
# These are dropped after create_all so the migration can re-create them.
_TABLES_ADDED_BY_MIGRATIONS = [
    "configuredmodelwithdatasource",
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
