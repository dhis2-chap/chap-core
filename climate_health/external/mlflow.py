from pathlib import Path
from typing import Generic, TypeVar
import logging
import pandas as pd
import mlflow

logger = logging.getLogger(__name__)
from climate_health.dataset import IsSpatioTemporalDataSet

FeatureType = TypeVar('FeatureType')

class ExternalMLflowModel(Generic[FeatureType]):
    """
    Wrapper around an mlflow model with commands for training and predicting
    """

    def __init__(self, model_path: str, adapters=None, working_dir="./"):
        self.model_path = model_path
        self._adapters = adapters
        self._working_dir = working_dir
        self._location_mapping = None
        self._model_file_name = Path(model_path).name + ".model"

    def train(self, train_data: IsSpatioTemporalDataSet[FeatureType], extra_args=None):
        if extra_args is None:
            extra_args = ''
        train_file_name = 'training_data.csv'
        train_file_name = Path(self._working_dir) / Path(train_file_name)
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
                                           "model_output_file": str(self._model_file_name)
                                       })
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


