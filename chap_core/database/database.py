import dataclasses
import time

import pandas as pd
from sqlmodel import SQLModel, create_engine, Session
from .tables import BackTest, BackTestForecast, Observation, DataSet, DebugEntry
# CHeck if CHAP_DATABASE_URL is set in the environment
import os

from chap_core.time_period import TimePeriod
from ..spatio_temporal_data.temporal_dataclass import DataSet as _DataSet

engine = None
database_url = os.getenv("CHAP_DATABASE_URL", default=None)
if database_url is not None:
    engine = create_engine(database_url)


class SessionWrapper:
    def __init__(self, local_engine=None, session=None):
        self.engine = local_engine#  or engine
        self.session = session

    def __enter__(self):
        self.session = Session(self.engine)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        return False

    def add_evaluation_results(self, evaluation_results, last_train_period: TimePeriod, dataset_id, model_id):
        backtest = BackTest(dataset_id=dataset_id, estimator_id=model_id, last_train_period_id=last_train_period.id)
        self.session.add(backtest)
        for eval_result in evaluation_results:
            first_period: TimePeriod = eval_result.period_range[0]
            for location, samples in eval_result.items():
                for period, value in zip(eval_result.period_range, samples.samples):
                    forecast = BackTestForecast(period_id=period.id, region_id=location,
                                                last_train_period_id=last_train_period.id,
                                                last_seen_period_id=first_period.id, values=value.tolist())
                    backtest.forecasts.append(forecast)
        self.session.commit()
        return backtest.id

    def add_dataset(self, dataset_name, orig_dataset: _DataSet, polygons):
        dataset = DataSet(name=dataset_name, polygons=polygons)
        for location, data in orig_dataset.items():
            field_names = [field.name for field in dataclasses.fields(data) if field.name not in ["time_period", "location"]]
            for row in data:
                for field in field_names:
                    observation = Observation(period_id=row.time_period.id, region_id=location, value=float(getattr(row, field)),
                                              element_id=field)
                    dataset.observations.append(observation)
        self.session.add(dataset)
        self.session.commit()
        return dataset.id

    def get_dataset(self, dataset_id, dataclass: type) -> _DataSet:
        # field_names = [field.name for field in dataclasses.fields(dataclass) if field.name not in ["time_period", "location"]]
        dataset = self.session.get(DataSet, dataset_id)
        observations = dataset.observations
        dataframe = pd.DataFrame([obs.dict() for obs in observations]).rename(columns={'region_id': 'location', 'period_id': 'time_period'})
        dataframe = dataframe.set_index(["location", "time_period"])
        pivoted = dataframe.pivot(columns="element_id", values="value").reset_index()
        dataset = _DataSet.from_pandas(pivoted, dataclass)
        return dataset

    def add_debug(self):
        ''' Function for debuginng'''
        debug_entry = DebugEntry(timestamp=time.time())
        self.session.add(debug_entry)
        self.session.commit()
        return debug_entry.id


def create_db_and_tables():
    if engine is not None:
        SQLModel.metadata.create_all(engine)

#create_db_and_tables()
