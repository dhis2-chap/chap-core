import dataclasses
import datetime
import json
import logging

# CHeck if CHAP_DATABASE_URL is set in the environment
import os
from pathlib import Path
import time
from typing import List, Optional

import psycopg2
import sqlalchemy
from sqlalchemy.orm import selectinload
from sqlmodel import Session, SQLModel, create_engine, select

from chap_core.datatypes import FullData, create_tsdataclass
from chap_core.geometry import Polygons
from chap_core.log_config import is_debug_mode
from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.time_period import Month, Week

from .. import ModelTemplateInterface
from ..external.model_configuration import ModelTemplateConfigV2
from ..models import ModelTemplate
from ..models.configured_model import ConfiguredModel
from ..models.external_chapkit_model import ExternalChapkitModelTemplate
from ..spatio_temporal_data.converters import observations_to_dataset
from ..spatio_temporal_data.temporal_dataclass import DataSet as _DataSet
from .dataset_tables import DataSet, Observation, DataSetCreateInfo, DataSetInfo
from .debug import DebugEntry
from .model_spec_tables import ModelSpecRead
from .model_templates_and_config_tables import ConfiguredModelDB, ModelConfiguration, ModelTemplateDB
from .tables import BackTest, Prediction, PredictionSamplesEntry

logger = logging.getLogger(__name__)
engine = None
database_url = os.getenv("CHAP_DATABASE_URL", default=None)
logger.info(f"Database url: {database_url}")
if database_url is not None:
    n = 0
    while n < 30:
        try:
            engine = create_engine(database_url, echo=is_debug_mode())
            break
        except sqlalchemy.exc.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}. Trying again")
            n += 1
            time.sleep(1)
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}. Trying again")
            n += 1
            time.sleep(1)
    else:
        raise ValueError("Failed to connect to database")
else:
    logger.warning("Database url not set. Database operations will not work")


class SessionWrapper:
    """
    This is a wrapper around data access operations.
    This class handles cases when putting things in/out of db requires
    more than just adding/getting a row, e.g. transforming data etc.
    """

    def __init__(self, local_engine=None, session=None):
        self.engine = local_engine  #  or engine
        self.session: Optional[Session] = session

    def __enter__(self):
        self.session = Session(self.engine)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        return False

    def list_all(self, model):
        return self.session.exec(select(model)).all()

    def create_if_not_exists(self, model, id_name="id"):
        logging.info(f"Create if not exist with {model}")
        T = type(model)
        if not self.session.exec(select(T).where(getattr(T, id_name) == getattr(model, id_name))).first():
            self.session.add(model)
            self.session.commit()
        return model

    def add_model_template(self, model_template: ModelTemplateDB) -> int:
        # check if model template already exists
        existing_template = self.session.exec(
            select(ModelTemplateDB).where(ModelTemplateDB.name == model_template.name)
        ).first()
        if existing_template:
            logger.info(f"Model template with name {model_template.name} already exists. Returning existing id")
            return existing_template.id

        # add db entry
        logger.info(f"Adding model template: {model_template}")
        self.session.add(model_template)
        self.session.commit()

        # return id
        return model_template.id

    def add_model_template_from_yaml_config(self, model_template_config: ModelTemplateConfigV2) -> int:
        """Sets the ModelSpecRead a yaml string.
        Note that the yaml string is what's defined in a model template's MLProject file,
        so source_url will have to be added manually."""
        # TODO: maybe just use add_model_template and make sure to structure it correctly first
        # TODO: needs cleanup
        # TODO: existing check should probably use name instead of source url
        # parse yaml content as dict
        existing_template = self.session.exec(
            select(ModelTemplateDB).where(ModelTemplateDB.name == model_template_config.name)
        ).first()

        d = model_template_config.model_dump()
        info = d.pop("meta_data")
        d = d | info

        if existing_template:
            logger.info(f"Model template with name {model_template_config.name} already exists. Updating it")
            # Update the existing template with new data
            for key, value in d.items():
                if hasattr(existing_template, key):
                    setattr(existing_template, key, value)
            self.session.commit()
            return existing_template.id

        # Create new template
        db_object = ModelTemplateDB(**d)
        logger.info(f"Adding model template: {db_object}")
        self.session.add(db_object)
        self.session.commit()
        return db_object.id

    def add_configured_model(
        self,
        model_template_id: int,
        configuration: ModelConfiguration,
        configuration_name="default",
        uses_chapkit=False,
    ) -> int:
        # get model template name
        model_template = self.session.exec(
            select(ModelTemplateDB).where(ModelTemplateDB.id == model_template_id)
        ).first()
        template_name = model_template.name

        # set configured name
        if configuration_name == "default":
            # default configurations just use the name of their model template (for backwards compatibility)
            name = template_name
        else:
            # combine model template with configuration name to make the name unique
            name = f"{template_name}:{configuration_name}"

        # check if configured model already exists
        existing_configured = self.session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.name == name)).first()
        if existing_configured:
            logger.info(f"Configured model with name {name} already exists. Returning existing id")
            return existing_configured.id

        # create and add db entry
        configured_model = ConfiguredModelDB(
            name=name,
            model_template_id=model_template_id,
            **configuration.model_dump(),
            model_template=model_template,
            uses_chapkit=uses_chapkit,
        )
        configured_model.validate_user_options(configured_model)
        # configured_model.validate_user_options(model_template)
        logger.info(f"Adding configured model: {configured_model}")
        self.session.add(configured_model)
        self.session.commit()

        # return id
        return configured_model.id

    def get_configured_models(self) -> List[ModelSpecRead]:
        # TODO: using ModelSpecRead for backwards compatibility, should in future return ConfiguredModelDB?

        # get configured models from db
        # configured_models = SessionWrapper(session=session).list_all(ConfiguredModelDB)
        configured_models = self.session.exec(
            select(ConfiguredModelDB).options(selectinload(ConfiguredModelDB.model_template))
        ).all()

        # serialize to json and combine configured model with model template
        configured_models_data = []
        for configured_model in configured_models:
            # get configured model and model template json data
            try:
                configured_data = configured_model.model_dump(mode="json")
                template_data = configured_model.model_template.model_dump(mode="json")

                # Debug logging for enum handling
                logger.debug(f"Processing configured model {configured_model.id}: {configured_model.name}")
                logger.debug(f"Template supported_period_type: {configured_model.model_template.supported_period_type}")

            except Exception as e:
                logger.error(
                    f"Error dumping model data for configured_model id={configured_model.id}, name={configured_model.name}"
                )
                logger.error(
                    f"Template id={configured_model.model_template.id if configured_model.model_template else 'None'}"
                )
                logger.error(f"Exception: {type(e).__name__}: {str(e)}")
                logger.error("Full traceback:", exc_info=True)
                raise

            # add display name for configuration (not stored in db)
            # stitch together template displayName with configured name stub
            template_display_name = configured_model.model_template.display_name
            if ":" in configured_model.name:
                # configured model name is already stitched together as template_name:configuration_name
                configuration_stub = configured_model.name.split(":")[-1]
                # combine model template with configuration name to make the name unique
                configuration_display_name = configuration_stub.replace("_", " ").capitalize()
                display_name = f"{template_display_name} [{configuration_display_name}]"
            else:
                # default configurations just use the display name of their model template
                display_name = template_display_name
            configured_data["display_name"] = display_name

            # merge json data and add to results
            # NOTE: the sequence is important, starting with template data and add/overwrite with configured model data
            # ...in case of conflicting attrs, eg id and name
            merged_data = {**template_data, **configured_data}
            configured_models_data.append(merged_data)

        # debug
        # import json
        # for m in configured_models_data:
        #    logger.info('list model data: ' + json.dumps(m, indent=4))

        # temp: convert to ModelSpecRead to preserve existing results
        # TODO: remove ModelSpecRead and return directly as ConfiguredModelDB
        for model in configured_models_data:
            # convert single target value to target dict
            model["target"] = {
                "name": model["target"],
                "displayName": model["target"].replace("_", " ").capitalize(),
                "description": model["target"].replace("_", " ").capitalize(),
            }
            # convert list of required covarate strings to list of covariate dicts
            model["covariates"] = [
                {
                    "name": cov,
                    "displayName": cov.replace("_", " ").capitalize(),
                    "description": cov.replace("_", " ").capitalize(),
                }
                for cov in model["required_covariates"]
            ]
            # add list of additional covariate strings to list of covariate dicts
            # Use .get() with default empty list for backwards compatibility with v1.0.17
            # Extract existing covariate names to avoid dict comparison issues
            existing_cov_names = [c["name"] for c in model["covariates"]]
            model["covariates"] += [
                {
                    "name": cov,
                    "displayName": cov.replace("_", " ").capitalize(),
                    "description": cov.replace("_", " ").capitalize(),
                }
                for cov in model.get("additional_continuous_covariates", [])
                if cov not in existing_cov_names
            ]
            model["archived"] = model.get("archived", False)
            model["uses_chapkit"] = model.get("uses_chapkit", False)
        # for m in configured_models_data:
        #    logger.info('converted list model data: ' + json.dumps(m, indent=4))
        configured_models_read = [ModelSpecRead.model_validate(m) for m in configured_models_data]
        # for m in configured_models_read:
        #    logger.info('read list model data: ' + json.dumps(m.model_dump(mode='json'), indent=4))

        # return
        return configured_models_read

    def get_configured_model_by_name(self, configured_model_name: str) -> ConfiguredModelDB:
        try:
            configured_model = self.session.exec(
                select(ConfiguredModelDB).where(ConfiguredModelDB.name == configured_model_name)
            ).one()
        except sqlalchemy.exc.NoResultFound:
            all_names = self.session.exec(select(ConfiguredModelDB.name)).all()
            raise ValueError(
                f"Configured model with name {configured_model_name} not found. Available names: {all_names}"
            )

        return configured_model

    def get_configured_model_with_code(self, configured_model_id: int) -> ConfiguredModel:
        logger.info(f"Getting configured model with id {configured_model_id}")
        configured_model = self.session.get(ConfiguredModelDB, configured_model_id)
        if configured_model.name == "naive_model":
            return NaiveEstimator()
        template_name = configured_model.model_template.name
        logger.info(f"Configured model: {configured_model}, template: {configured_model.model_template}")
        ignore_env = (
            template_name.startswith("chap_ewars") or template_name == "ewars_template"
        )  # TODO: seems hacky, how to fix?

        if configured_model.uses_chapkit:
            logger.info(f"Assuming chapkit model at {configured_model.model_template.source_url}")
            template = ExternalChapkitModelTemplate(configured_model.model_template.source_url)
            logger.info(f"template: {template}")
            logger.info(f"configured_model: {configured_model}")
            return template.get_model(configured_model)
        else:
            logger.info(f"Assuming github model at {configured_model.model_template.source_url}")
            return ModelTemplate.from_directory_or_github_url(
                configured_model.model_template.source_url,
                ignore_env=ignore_env,
            ).get_model(configured_model)

    def get_model_template(self, model_template_id: int) -> ModelTemplateInterface:
        model_template = self.session.get(ModelTemplateDB, model_template_id)
        if model_template is None:
            raise ValueError(f"Model template with id {model_template_id} not found")
        return model_template

    def get_backtest_with_truth(self, backtest_id: int) -> BackTest:
        backtest = self.session.get(BackTest, backtest_id)
        if backtest is None:
            raise ValueError(f"Backtest with id {backtest_id} not found")
        dataset = backtest.dataset
        if dataset is None:
            raise ValueError(f"Dataset for backtest with id {backtest_id} not found")
        entries = backtest.forecasts
        if entries is None or len(entries) == 0:
            raise ValueError(f"No forecasts found for backtest with id {backtest_id}")

    def add_backtest(self, backtest: BackTest) -> None:
        self.session.add(backtest)
        self.session.commit()

    def add_predictions(self, predictions, dataset_id, model_id, name, metadata: dict = {}):
        n_periods = len(list(predictions.values())[0])
        samples_ = [
            PredictionSamplesEntry(period=period.id, org_unit=location, values=value.tolist())
            for location, data in predictions.items()
            for period, value in zip(data.time_period, data.samples)
        ]
        org_units = list(predictions.keys())
        model_db_id = self.session.exec(select(ConfiguredModelDB.id).where(ConfiguredModelDB.name == model_id)).first()

        prediction = Prediction(
            dataset_id=dataset_id,
            model_id=model_id,
            name=name,
            created=datetime.datetime.now(),
            n_periods=n_periods,
            meta_data=metadata,
            forecasts=samples_,
            org_units=org_units,
            model_db_id=model_db_id,
        )
        self.session.add(prediction)
        self.session.commit()
        return prediction.id

    def add_dataset_from_csv(self, name: str, csv_path: Path, geojson_path: Optional[Path] = None):
        dataset = _DataSet.from_csv(csv_path, dataclass=FullData)
        geojson_content = open(geojson_path, "r").read() if geojson_path else None
        features = None
        if geojson_content is not None:
            features = Polygons.from_geojson(json.loads(geojson_content), id_property="NAME_1").feature_collection()
            features = features.model_dump_json()

        return self.add_dataset(DataSetCreateInfo(name=name), dataset, features)

    def add_dataset(self, dataset_info: DataSetCreateInfo, orig_dataset: _DataSet, polygons):
        """
        Add a dataset to the database. The dataset is provided as a spatio-temporal dataclass.
        The polygons should be provided as a geojson feature collection.
        The dataset_info should contain information about the dataset, such as its name and data sources.
        The function sets some derived fields in the dataset_info, such as the first and last time period and the covariates.
        The function returns the id of the newly created dataset.
        """
        logger.info(
            f"Adding dataset {dataset_info.name} with {len(list(orig_dataset.locations()))} locations and {len(orig_dataset.period_range)} time periods"
        )
        field_names = [
            field.name
            for field in dataclasses.fields(next(iter(orig_dataset.values())))
            if field.name not in ["time_period", "location"]
        ]
        logger.info(f"Field names in dataset: {field_names}")
        if isinstance(orig_dataset.period_range[0], Month):
            period_type = "month"
        else:
            assert isinstance(orig_dataset.period_range[0], Week), orig_dataset.period_range[0]
            period_type = "week"
        full_info = DataSetInfo(
            first_period=orig_dataset.period_range[0].id,
            last_period=orig_dataset.period_range[-1].id,
            covariates=field_names,
            created=datetime.datetime.now(),
            org_units=list(orig_dataset.locations()),
            period_type=period_type,
            **dataset_info.model_dump(),
        )
        dataset = DataSet(geojson=polygons, **full_info.model_dump())

        for location, data in orig_dataset.items():
            field_names = [
                field.name for field in dataclasses.fields(data) if field.name not in ["time_period", "location"]
            ]
            for row in data:
                for field in field_names:
                    observation = Observation(
                        period=row.time_period.id,
                        org_unit=location,
                        value=float(getattr(row, field)),
                        feature_name=field,
                    )
                    dataset.observations.append(observation)

        self.session.add(dataset)
        self.session.commit()
        assert self.session.exec(select(Observation).where(Observation.dataset_id == dataset.id)).first() is not None
        return dataset.id

    def get_dataset(self, dataset_id, dataclass: type | None = None) -> _DataSet:
        dataset = self.session.get(DataSet, dataset_id)
        if dataclass is None:
            logger.info(f"Getting dataset with covariates: {dataset.covariates} and name: {dataset.name}")
            field_names = dataset.covariates
            dataclass = create_tsdataclass(field_names)
        observations = dataset.observations
        new_dataset = observations_to_dataset(dataclass, observations)

        if dataset.geojson:
            logger.info(f"Loading polygons from geojson for dataset id {dataset_id}")
            new_dataset.set_polygons(Polygons.from_geojson(json.loads(dataset.geojson), id_property="district").data)

        return new_dataset

    def get_dataset_by_name(self, dataset_name: str) -> Optional[DataSet]:
        dataset = self.session.exec(select(DataSet).where(DataSet.name == dataset_name)).first()
        return dataset

    def add_debug(self):
        """Function for debuging"""
        debug_entry = DebugEntry(timestamp=time.time())
        self.session.add(debug_entry)
        self.session.commit()
        return debug_entry.id


def _run_alembic_migrations(engine):
    """
    Run Alembic migrations programmatically.
    This is called after the custom migration system to apply any Alembic-managed schema changes.
    """
    from alembic import command
    from alembic.config import Config

    logger.info("Running Alembic migrations")

    try:
        # Get the path to alembic.ini relative to the project root
        project_root = Path(__file__).parent.parent.parent
        alembic_ini_path = project_root / "alembic.ini"

        if not alembic_ini_path.exists():
            logger.warning(f"Alembic config not found at {alembic_ini_path}. Skipping Alembic migrations.")
            return

        # Create Alembic config
        alembic_cfg = Config(str(alembic_ini_path))

        # Pass the engine connection to Alembic for programmatic usage
        with engine.connect() as connection:
            alembic_cfg.attributes["connection"] = connection
            command.upgrade(alembic_cfg, "head")

        logger.info("Completed Alembic migrations successfully")

    except Exception as e:
        logger.error(f"Error during Alembic migrations: {e}", exc_info=True)
        # Don't raise - allow system to continue if Alembic fails
        # This ensures backward compatibility


def create_db_and_tables():
    # TODO: Read config for options on how to create the database migrate/update/seed/seed_and_update
    if engine is not None:
        logger.info("Engine set. Creating tables")
        n = 0
        while n < 30:
            try:
                # Step 1: Run custom migrations for backward compatibility (v1.0.17, etc.)
                _run_generic_migration(engine)

                # Step 2: Create any new tables that don't exist yet
                SQLModel.metadata.create_all(engine)

                # Step 3: Run Alembic migrations for future schema changes
                _run_alembic_migrations(engine)

                logger.info("Table created and migrations completed")
                break
            except sqlalchemy.exc.OperationalError as e:
                logger.error(f"Failed to create tables: {e}. Trying again")
                n += 1
                time.sleep(1)
                if n >= 20:
                    raise e
        with SessionWrapper(engine) as session:
            from .model_template_seed import seed_configured_models_from_config_dir

            seed_configured_models_from_config_dir(session.session)
            # seed_example_datasets(session)
    else:
        logger.warning("Engine not set. Tables not created")


def _run_v1_0_17_migrations(conn, engine):
    """
    Specific migrations needed when upgrading from v1.0.17 to current version.
    This handles data type conversions and corrections that the generic migration cannot handle.
    """
    logger.info("Running v1.0.17 specific migrations")

    inspector = sqlalchemy.inspect(engine)
    existing_tables = inspector.get_table_names()

    try:
        # Fix: Check if modeltemplatedb table exists and has corrupted data
        if "modeltemplatedb" in existing_tables:
            logger.info("Checking for corrupted PeriodType enum values in modeltemplatedb")

            # Check if there are any rows with invalid enum values (like '1' instead of 'week', 'month', etc.)
            check_sql = """
                SELECT COUNT(*) as count
                FROM modeltemplatedb
                WHERE supported_period_type NOT IN ('week', 'month', 'any', 'year')
            """
            result = conn.execute(sqlalchemy.text(check_sql)).fetchone()

            if result and result[0] > 0:
                logger.warning(f"Found {result[0]} rows with corrupted PeriodType enum values, fixing...")

                # Map common corrupted values to correct enum values
                # If the value is '1', we'll default to 'any' as it's the most permissive
                fix_sql = """
                    UPDATE modeltemplatedb
                    SET supported_period_type = 'any'::periodtype
                    WHERE supported_period_type NOT IN ('week', 'month', 'any', 'year')
                """
                conn.execute(sqlalchemy.text(fix_sql))
                conn.commit()
                logger.info("Fixed corrupted PeriodType enum values")

        # Fix: Ensure JSON columns that should be arrays are arrays, not objects
        if "dataset" in existing_tables:
            columns = {col["name"] for col in inspector.get_columns("dataset")}

            # Fix data_sources if it exists and contains objects instead of arrays
            if "data_sources" in columns:
                logger.info("Fixing data_sources column in dataset table")
                fix_sql = """
                    UPDATE dataset
                    SET data_sources = '[]'::json
                    WHERE data_sources IS NULL OR data_sources::text = '{}'
                """
                conn.execute(sqlalchemy.text(fix_sql))
                conn.commit()

            # Fix org_units if it exists and contains objects instead of arrays
            if "org_units" in columns:
                logger.info("Fixing org_units column in dataset table")
                fix_sql = """
                    UPDATE dataset
                    SET org_units = '[]'::json
                    WHERE org_units IS NULL OR org_units::text = '{}'
                """
                conn.execute(sqlalchemy.text(fix_sql))
                conn.commit()

            # Fix covariates if it contains objects instead of arrays
            if "covariates" in columns:
                logger.info("Fixing covariates column in dataset table")
                fix_sql = """
                    UPDATE dataset
                    SET covariates = '[]'::json
                    WHERE covariates IS NULL OR covariates::text = '{}'
                """
                conn.execute(sqlalchemy.text(fix_sql))
                conn.commit()

        # Fix backtest table JSON columns
        if "backtest" in existing_tables:
            columns = {col["name"] for col in inspector.get_columns("backtest")}

            if "org_units" in columns:
                logger.info("Fixing org_units column in backtest table")
                fix_sql = """
                    UPDATE backtest
                    SET org_units = '[]'::json
                    WHERE org_units IS NULL OR org_units::text = '{}'
                """
                conn.execute(sqlalchemy.text(fix_sql))
                conn.commit()

            if "split_periods" in columns:
                logger.info("Fixing split_periods column in backtest table")
                fix_sql = """
                    UPDATE backtest
                    SET split_periods = '[]'::json
                    WHERE split_periods IS NULL OR split_periods::text = '{}'
                """
                conn.execute(sqlalchemy.text(fix_sql))
                conn.commit()

        # Fix configuredmodeldb table JSON columns
        if "configuredmodeldb" in existing_tables:
            columns = {col["name"] for col in inspector.get_columns("configuredmodeldb")}

            if "user_option_values" in columns:
                logger.info("Fixing user_option_values column in configuredmodeldb table")
                # This one should actually be an object {}, not an array
                fix_sql = """
                    UPDATE configuredmodeldb
                    SET user_option_values = '{}'::json
                    WHERE user_option_values IS NULL
                """
                conn.execute(sqlalchemy.text(fix_sql))
                conn.commit()

            if "additional_continuous_covariates" in columns:
                logger.info("Fixing additional_continuous_covariates column in configuredmodeldb table")
                fix_sql = """
                    UPDATE configuredmodeldb
                    SET additional_continuous_covariates = '[]'::json
                    WHERE additional_continuous_covariates IS NULL OR additional_continuous_covariates::text = '{}'
                """
                conn.execute(sqlalchemy.text(fix_sql))
                conn.commit()

        logger.info("Completed v1.0.17 specific migrations successfully")

    except Exception as e:
        logger.error(f"Error during v1.0.17 migrations: {e}")
        conn.rollback()
        raise


def _run_generic_migration(engine):
    """
    Generic migration function that adds missing columns to existing tables
    and sets default values for new columns in existing records.
    """
    logger.info("Running generic migration for missing columns")

    with engine.connect() as conn:
        # Run v1.0.17 specific migrations first
        # _run_v1_0_17_migrations(conn, engine)

        # Get current database schema
        inspector = sqlalchemy.inspect(engine)
        existing_tables = inspector.get_table_names()

        for table_name, table in SQLModel.metadata.tables.items():
            if table_name not in existing_tables:
                logger.info(f"Table {table_name} doesn't exist yet, will be created")
                continue

            # Get existing columns in the database
            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

            # Check for missing columns
            for column in table.columns:
                if column.name not in existing_columns:
                    logger.info(f"Adding missing column {column.name} to table {table_name}")

                    # Add the column
                    column_type = column.type.compile(dialect=engine.dialect)
                    add_column_sql = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {column_type}"

                    try:
                        conn.execute(sqlalchemy.text(add_column_sql))

                        # Set default value based on column type and properties
                        default_value = _get_column_default_value(column)
                        if default_value is not None:
                            update_sql = (
                                f"UPDATE {table_name} SET {column.name} = :default_val WHERE {column.name} IS NULL"
                            )
                            conn.execute(sqlalchemy.text(update_sql), {"default_val": default_value})
                            logger.info(f"Set default value for {column.name} in existing records")

                        conn.commit()
                        logger.info(f"Successfully added column {column.name} to {table_name}")

                    except Exception as e:
                        logger.error(f"Failed to add column {column.name} to {table_name}: {e}")
                        conn.rollback()


def _get_column_default_value(column):
    """
    Determine appropriate default value for a column based on its type and properties.
    """
    # Check if column has a default value defined
    if column.default is not None:
        if hasattr(column.default, "arg") and column.default.arg is not None:
            # Check if it's a factory function (like list or dict)
            if callable(column.default.arg):
                try:
                    result = column.default.arg()
                    if isinstance(result, list):
                        return "[]"
                    elif isinstance(result, dict):
                        return "{}"
                except Exception:
                    pass
            return column.default.arg

    # Check column type and provide appropriate defaults
    column_type = str(column.type).lower()

    if "json" in column_type or "pydanticlisttype" in column_type:
        # For JSON columns, only default to [] if there's a default_factory set
        # If there's no explicit default, use NULL (safer for Optional[dict] fields)
        # Most list JSON columns have default_factory=list which is handled above
        return None
    elif "varchar" in column_type or "text" in column_type:
        return ""  # Empty string for text columns
    elif "integer" in column_type or "numeric" in column_type:
        return 0  # Zero for numeric columns
    elif "boolean" in column_type:
        return False  # False for boolean columns
    elif "timestamp" in column_type or "datetime" in column_type:
        return None  # Let NULL remain for timestamps

    return None  # Default to NULL for unknown types
