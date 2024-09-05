from pathlib import Path
from typing import Generic, TypeVar
import logging
import numpy as np
import pandas
import pandas as pd
import mlflow

from climate_health.datatypes import SummaryStatistics, HealthData
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet
from climate_health.time_period import TimePeriod

logger = logging.getLogger(__name__)

FeatureType = TypeVar('FeatureType')

class ExternalMLflowModel(Generic[FeatureType]):
    """
    Wrapper around an mlflow model with commands for training and predicting
    """

    def __init__(self, model_path: str, name: str=None, adapters=None, working_dir="./", data_type=HealthData):
        self.model_path = model_path
        self._adapters = adapters
        self._working_dir = working_dir
        self._location_mapping = None
        self._model_file_name = Path(model_path).name + ".model"
        self.is_lagged = True
        self._data_type = data_type
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self):
        return self

    def train(self, train_data: DataSet, extra_args=None):

        if extra_args is None:
            extra_args = ''

        train_file_name = 'training_data.csv'
        #train_file_name = Path(self._working_dir) / Path(train_file_name)
        pd = train_data.to_pandas()
        new_pd = self._adapt_data(pd)
        new_pd.to_csv(train_file_name)
        print(train_file_name)

        # touch model output file
        with open(self._model_file_name, 'w') as f:
            pass

        response = mlflow.projects.run(str(self.model_path), entry_point="train",
                                       parameters={
                                           "train_data": str(train_file_name),
                                           "model": str(self._model_file_name)
                                       },
                                       build_image=True)
        self._saved_state = new_pd
        print(response)

    def _adapt_data(self, data: pd.DataFrame, inverse=False):
        if self._location_mapping is not None:
            data['location'] = data['location'].apply(self._location_mapping.name_to_index)
        if self._adapters is None:
            return data
        adapters = self._adapters
        if inverse:
            adapters = {v: k for k, v in adapters.items()}
            # data['disease_cases'] = data[adapters['disase_cases']]
            return data

        for to_name, from_name in adapters.items():
            if from_name == 'week':
                if hasattr(data['time_period'], 'dt'):
                    new_val = data['time_period'].dt.week
                    data[to_name] = new_val
                else:
                    data[to_name] = [int(str(p).split('W')[-1]) for p in data['time_period']]  # .dt.week

            elif from_name == 'month':
                data[to_name] = data['time_period'].dt.month
            elif from_name == 'year':
                if hasattr(data['time_period'], 'dt'):
                    data[to_name] = data['time_period'].dt.year
                else:
                    data[to_name] = [int(str(p).split('W')[0]) for p in
                                     data['time_period']]  # data['time_period'].dt.year
            else:
                data[to_name] = data[from_name]
        return data

    def predict(self, future_data: DataSet) -> DataSet:
        name = 'future_data.csv'
        future_data_name = Path(self._working_dir) / Path(name)
        start_time = future_data.start_timestamp
        logger.info('Predicting on dataset from %s', start_time)
        with open(name, "w") as f:
            df = future_data.to_pandas()
            df['disease_cases'] = np.nan

            new_pd = self._adapt_data(df)
            if self.is_lagged:
                new_pd = pd.concat([self._saved_state, new_pd]).sort_values(['location', 'time_period'])
            new_pd.to_csv(future_data_name)

        #command = self._predict_command.format(future_data=name,
        #                                       model=self._model_file_name,
        #                                       out_file='predictions.csv', **kwargs)
        #response = self.run_through_container(command)

        predictions_file = Path(self._working_dir) / 'predictions.csv'
        # touch predictions.csv
        with open(predictions_file, 'w') as f:
            pass

        response = mlflow.projects.run(str(self.model_path), entry_point="predict",
                                        parameters={
                                             "future_data": str(future_data_name), "model": str(self._model_file_name),
                                             "out_file": str(predictions_file)
                                        })
        try:
            df = pd.read_csv(predictions_file)

        except pandas.errors.EmptyDataError:
            # todo: Probably deal with this in an other way, throw an exception istead
            logging.warning("No data returned from model (empty file from predictions)")
            raise ValueError(f"No prediction data written")
        result_class = SummaryStatistics if 'quantile_low' in df.columns else HealthData
        if self._location_mapping is not None:
            df['location'] = df['location'].apply(self._location_mapping.index_to_name)

        time_periods = [TimePeriod.parse(s) for s in df.time_period.astype(str)]
        mask = [start_time <= time_period.start_timestamp for time_period in time_periods]
        df = df[mask]
        return DataSet.from_pandas(df, result_class)

