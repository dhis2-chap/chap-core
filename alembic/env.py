import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all SQLModel models to ensure they are registered with metadata
# This is required for autogenerate to detect all tables and columns
from chap_core.database.dataset_tables import DataSet, Observation
from chap_core.database.debug import DebugEntry
from chap_core.database.feature_tables import FeatureSource, FeatureType
from chap_core.database.model_spec_tables import ModelFeatureLink, ModelSpec
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.database.tables import BackTest, BackTestForecast, BackTestMetric, Prediction, PredictionSamplesEntry

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = SQLModel.metadata

# Get database URL from environment variable, with default for local development
database_url = os.getenv("CHAP_DATABASE_URL", "postgresql://root:thisisnotgoingtobeexposed@localhost:5432/chap_core")
config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Compare type is needed for proper column type comparison
        compare_type=True,
        # Compare server default is needed for detecting default value changes
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Check if a connection is already provided (for programmatic usage)
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        # Create engine from config if no connection is provided
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    if connectable is None:
        raise ValueError("No database connection available. Set CHAP_DATABASE_URL environment variable.")

    # Check if connectable is already a Connection or an Engine
    if hasattr(connectable, 'connect'):
        # It's an Engine, need to create a connection
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                # Compare type is needed for proper column type comparison
                compare_type=True,
                # Compare server default is needed for detecting default value changes
                compare_server_default=True,
            )

            with context.begin_transaction():
                context.run_migrations()
    else:
        # It's already a Connection, use it directly
        context.configure(
            connection=connectable,
            target_metadata=target_metadata,
            # Compare type is needed for proper column type comparison
            compare_type=True,
            # Compare server default is needed for detecting default value changes
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
