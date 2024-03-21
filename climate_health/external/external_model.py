import logging
import subprocess
import tempfile
from typing import Protocol, Generic, TypeVar

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData


class IsExternalModel(Protocol):
    def get_predictions(self, train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: IsSpatioTemporalDataSet[ClimateData]) -> IsSpatioTemporalDataSet[HealthData]:
        ...


FeatureType = TypeVar('FeatureType')


class ExternalCommandLineModel(Generic[FeatureType]):
    def __init__(self,
                 setup_command: str,
                 train_command: str,
                 predict_command: str,
                 data_type: type[FeatureType],
                 conda_env_file: str):
        self._train_command = train_command
        self._predict_command = predict_command
        self._data_type = data_type
        self._conda_env_file = conda_env_file
        self._model = None

    def setup(self):
        self._run_command(self._setup_command)

    def train(self, train_data: IsSpatioTemporalDataSet[FeatureType]):
        self._out_file = tempfile.NamedTemporaryFile()
        out_file_name = self._out_file.name
        self._saved_state = train_data

        with tempfile.NamedTemporaryFile() as train_datafile:
            train_data.to_csv(train_datafile.name)
            command = self._train_command.format(train_data=train_datafile.name, model=out_file_name)
            response = run_command(command)

    def predict(self, future_data: IsSpatioTemporalDataSet[FeatureType]) -> IsSpatioTemporalDataSet[FeatureType]:
        with tempfile.NamedTemporaryFile() as future_datafile:
            future_data.to_csv(future_datafile.name)
            command = self._predict_command.format(future_data=future_datafile.name, model=self._out_file.name)
            response = run_command(command)
            return response

def run_command(command: str):

    """Runs a unix command using subprocess"""
    command = command.split()
    try:
        output = subprocess.check_output(command)
        logging.info(output)
    except subprocess.CalledProcessError as e:
        error = e.output.decode()
        logging.info(error)
        raise e

    return output
