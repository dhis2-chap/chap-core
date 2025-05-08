import dataclasses
import datetime
import time
from typing import Optional

import psycopg2
import sqlalchemy
from sqlmodel import SQLModel, create_engine, Session, select
from .tables import BackTest, BackTestForecast, Prediction, PredictionSamplesEntry
from .model_spec_tables import seed_with_session_wrapper
from .debug import DebugEntry
from .dataset_tables import Observation, DataSet
# CHeck if CHAP_DATABASE_URL is set in the environment
import os

from chap_core.time_period import TimePeriod
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

    def add_evaluation_results(self, evaluation_results, last_train_period: TimePeriod, info: BackTestCreate):
        info.created = datetime.datetime.now()
        backtest = BackTest(last_train_period=last_train_period.id,
                            **info.dict())
        self.session.add(backtest)
        for eval_result in evaluation_results:
            first_period: TimePeriod = eval_result.period_range[0]
            for location, samples in eval_result.items():
                for period, value in zip(eval_result.period_range, samples.samples):
                    forecast = BackTestForecast(period=period.id, org_unit=location,
                                                last_train_period=last_train_period.id,
                                                last_seen_period=first_period.id, values=value.tolist())
                    backtest.forecasts.append(forecast)
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


