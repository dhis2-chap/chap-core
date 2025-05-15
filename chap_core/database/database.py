import dataclasses
import datetime
import time
from typing import Optional

import psycopg2
import sqlalchemy
from chap_core.predictor.naive_estimator import NaiveEstimator
from sqlmodel import SQLModel, create_engine, Session, select
from .tables import BackTest, BackTestForecast, Prediction, PredictionSamplesEntry
from .model_spec_tables import seed_with_session_wrapper
from .model_templates_and_config_tables import ModelTemplateDB, ConfiguredModelDB
from .debug import DebugEntry
from .dataset_tables import Observation, DataSet
# CHeck if CHAP_DATABASE_URL is set in the environment
import os

from chap_core.time_period import TimePeriod
from .. import ModelTemplateInterface
from ..external.model_configuration import ModelTemplateConfig
from ..models import ModelTemplate
from ..models.configured_model import ConfiguredModel
from ..rest_api_src.data_models import BackTestCreate
from ..spatio_temporal_data.converters import observations_to_dataset
from ..spatio_temporal_data.temporal_dataclass import DataSet as _DataSet
import logging
logger = logging.getLogger(__name__)
engine = None
database_url = os.getenv("CHAP_DATABASE_URL", default=None)
logger.info(f"Database url: {database_url}")
if database_url is not None:
    n = 0
    while n < 30:
        try:
            engine = create_engine(database_url)
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
    '''
    This is a wrapper around data access operations
    '''

    def __init__(self, local_engine=None, session=None):
        self.engine = local_engine#  or engine
        self.session: Optional[Session] = session

    def __enter__(self):
        self.session = Session(self.engine)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        return False

    def list_all(self, model):
        return self.session.exec(select(model)).all()

    def create_if_not_exists(self, model, id_name='id'):
        logging.info(f"Create if not exist with {model}")
        T = type(model)
        if not self.session.exec(select(T).where(getattr(T, id_name) == getattr(model, id_name))).first():
            self.session.add(model)
            self.session.commit()
        return model

    def add_model_template(self, model_template_config: ModelTemplateConfig) -> int:
        """Sets the ModelSpecRead a yaml string.
        Note that the yaml string is what's defined in a model template's MLProject file,
        so source_url will have to be added manually."""
        # parse yaml content as dict
        existing_template = self.session.exec(
            select(ModelTemplateDB).where(ModelTemplateDB.source_url == model_template_config.source_url)).first()
        if existing_template:
            return existing_template.id
        d = model_template_config.dict()
        info = d.pop('meta_data')
        d = d | info
        db_object = ModelTemplateDB(**d)

        self.session.add(db_object)
        self.session.commit()
        return db_object.id

    def add_configured_model(self, model_template_id: int, configuration: dict, name='default') -> int:
        existing_configured = self.session.exec(
            select(ConfiguredModelDB).where(ConfiguredModelDB.name == name).where(ConfiguredModelDB.model_template_id == model_template_id)).first()
        if existing_configured:
            logger.info(f"Configured model with name {name} already exists. Returning existing id")
            return existing_configured.id
        template_name = self.session.exec(
            select(ModelTemplateDB).where(ModelTemplateDB.id == model_template_id)).first().name
        configured_model = ConfiguredModelDB(name=f'{template_name}:{name}', model_template_id=model_template_id, configuration=configuration)
        self.session.add(configured_model)
        self.session.commit()
        return configured_model.id

    def get_configured_model(self, configured_model_id: int) -> ConfiguredModel:
        configured_model = self.session.get(ConfiguredModelDB, configured_model_id)
        if configured_model.name == 'naive_model':
            return NaiveEstimator()
        ignore_env = configured_model.model_template.name.startswith('chap_ewars')
        return ModelTemplate.from_directory_or_github_url(configured_model.model_template.source_url,
                                                          ignore_env=ignore_env,
                                                          ).get_model(configured_model.configuration)

    def get_model_template(self, model_template_id: int) -> ModelTemplateInterface:
        model_template =  self.session.get(ModelTemplateDB, model_template_id)
        if model_template is None:
            raise ValueError(f"Model template with id {model_template_id} not found")
        return model_template


    def add_evaluation_results(self, evaluation_results, last_train_period: TimePeriod, info: BackTestCreate):
        info.created = datetime.datetime.now()
        #org_units = list({location for ds in evaluation_results for location in ds.locations()})
        #split_points = list({er.period_range[0] for er in evaluation_results})
        backtest = BackTest(**info.dict())
        self.session.add(backtest)
        org_units = set([])
        split_points = set([])
        for eval_result in evaluation_results:
            first_period: TimePeriod = eval_result.period_range[0]
            split_points.add(first_period.id)
            for location, samples in eval_result.items():
                org_units.add(location)
                for period, value in zip(eval_result.period_range, samples.samples):
                    forecast = BackTestForecast(
                        period=period.id, org_unit=location,
                        last_train_period=last_train_period.id,
                        last_seen_period=first_period.id, values=value.tolist())
                    backtest.forecasts.append(forecast)
        backtest.org_units = list(org_units)
        backtest.split_periods = list(split_points)
        self.session.commit()
        return backtest.id

    def add_predictions(self, predictions, dataset_id, model_id, name, metadata: dict={}):
        n_periods = len(list(predictions.values())[0])
        prediction = Prediction(dataset_id=dataset_id,
                                model_id=model_id,
                                name=name,
                                created=datetime.datetime.now(),
                                n_periods=n_periods,
                                meta_data=metadata,
                                forecasts=[
                                    PredictionSamplesEntry(period=period.id,
                                                           org_unit=location,
                                                           values=value.tolist())
                                for location, data in predictions.items() for period, value in zip(data.time_period, data.samples)])
        self.session.add(prediction)
        self.session.commit()
        return prediction.id

    def add_dataset(self, dataset_name, orig_dataset: _DataSet, polygons, dataset_type: str | None = None):
        logger.info(f"Adding dataset {dataset_name} with {len(list(orig_dataset.locations()))} locations and {len(orig_dataset.period_range)} time periods")
        field_names = [field.name for field in dataclasses.fields(next(iter(orig_dataset.values()))) if field.name not in ["time_period", "location"]]
        dataset = DataSet(name=dataset_name, polygons=polygons, created=datetime.datetime.now(), covariates=field_names, type=dataset_type)
        for location, data in orig_dataset.items():
            field_names = [field.name for field in dataclasses.fields(data) if field.name not in ["time_period", "location"]]
            for row in data:
                for field in field_names:
                    observation = Observation(period=row.time_period.id, org_unit=location, value=float(getattr(row, field)),
                                              feature_name=field)
                    dataset.observations.append(observation)
        self.session.add(dataset)
        self.session.commit()
        assert self.session.exec(select(Observation).where(Observation.dataset_id==dataset.id)).first() is not None
        return dataset.id

    def get_dataset(self, dataset_id, dataclass: type) -> _DataSet:
        dataset = self.session.get(DataSet, dataset_id)
        observations = dataset.observations
        new_dataset = observations_to_dataset(dataclass, observations)
        return new_dataset

    def add_debug(self):
        ''' Function for debuging'''
        debug_entry = DebugEntry(timestamp=time.time())
        self.session.add(debug_entry)
        self.session.commit()
        return debug_entry.id


def create_db_and_tables():
    # TODO: Read config for options on how to create the database migrate/update/seed/seed_and_update
    if engine is not None:
        logger.info("Engin set. Creating tables")
        n = 0
        while n < 30:
            try:
                SQLModel.metadata.create_all(engine)
                break
            except sqlalchemy.exc.OperationalError as e:
                logger.error(f"Failed to create tables: {e}. Trying again")
                n += 1
                time.sleep(1)
        with SessionWrapper(engine) as session:
            seed_with_session_wrapper(session)
    else:
        logger.warning("Engine not set. Tables not created")


