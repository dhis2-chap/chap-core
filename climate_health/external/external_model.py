import logging
import subprocess
import tempfile
from typing import Protocol, Generic, TypeVar

import pandas.errors

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class IsExternalModel(Protocol):
    def get_predictions(self, train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: IsSpatioTemporalDataSet[ClimateData]) -> IsSpatioTemporalDataSet[HealthData]:
        ...


FeatureType = TypeVar('FeatureType')


class ExternalCommandLineModel(Generic[FeatureType]):
    """
    Represents a model with commands for setup (optional), training and prediction.
    Commands should contain placeholders for the train_data, future_data and model file,
    which are {train_data}, {future_data} and {model} respectively.

    The {model} is the file that the model is written to after training.

    conda_env_file can be a path to a  conda yml file that will be used to create a conda environment
    that everything will be run inside

    """
    def __init__(self,
                 name: str,
                 train_command: str,
                 predict_command: str,
                 data_type: type[FeatureType],
                 setup_command: str=None,
                 conda_env_file: str=None):
        self._name = name
        self._conda_env_name = "climate_health_" + self._name
        self._setup_command = setup_command
        self._train_command = train_command
        self._predict_command = predict_command
        self._data_type = data_type
        self._conda_env_file = conda_env_file
        self._model = None

    def run_through_conda(self, command: str):
        if self._conda_env_file:
            return f'conda run -n {self._conda_env_name} {command}'
        return run_command(command)

    def setup(self):
        if self._conda_env_file:
            try:
                run_command(f'conda env create --name {self._conda_env_name} -f {self._conda_env_file}')
            except subprocess.CalledProcessError:
                logging.info("Ignoring error when creating conda environment")
                pass

            self.run_through_conda(self._setup_command)
        elif self._setup_command is not None:
            run_command(self._setup_command)

    def deactivate(self):
        if self._conda_env_file:
            run_command(f'conda deactivate')

    def train(self, train_data: IsSpatioTemporalDataSet[FeatureType]):
        self._model_file_name = self._name + ".Rdata"   # todo: use tempfile or something else

        #with tempfile.NamedTemporaryFile() as train_datafile:
        train_file_name = "tmp_train.csv"
        with open(train_file_name, "w") as train_datafile:
            train_data.to_csv(train_file_name)
            command = self._train_command.format(train_data=train_file_name, model=self._model_file_name)
            response = self.run_through_conda(command)
            print(response)

    def predict(self, future_data: IsSpatioTemporalDataSet[FeatureType]) -> IsSpatioTemporalDataSet[FeatureType]:
        with tempfile.NamedTemporaryFile() as out_file:
            with tempfile.NamedTemporaryFile() as future_datafile:
                future_data.to_csv(future_datafile.name)
                command = self._predict_command.format(future_data=future_datafile.name,
                                                       model=self._model_file_name,
                                                       out_file=out_file.name)
                response = self.run_through_conda(command)
                try:
                    return SpatioTemporalDict.from_csv(out_file.name, self._data_type)
                except pandas.errors.EmptyDataError:
                    # todo: Probably deal with this in an other way, throw an exception istead
                    logging.warning("No data returned from model (empty file from predictions)")
                    return SpatioTemporalDict({})


def run_command(command: str):

    """Runs a unix command using subprocess"""
    logging.info(f"Running command: {command}")
    command = command.split()

    try:
        output = subprocess.check_output(command)
        logging.info(output)
    except subprocess.CalledProcessError as e:
        error = e.output.decode()
        logging.info(error)
        raise e

    return output
