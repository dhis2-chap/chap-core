import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol, Generic, TypeVar

import pandas as pd
import pandas.errors
import yaml

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict

logger = logging.getLogger(__name__)


class IsExternalModel(Protocol):
    def get_predictions(self, train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: IsSpatioTemporalDataSet[ClimateData]) -> IsSpatioTemporalDataSet[
        HealthData]:
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

    def __init__(self, name: str, train_command: str, predict_command: str, data_type: type[FeatureType],
                 setup_command: str = None, conda_env_file: str = None,
                 working_dir="./", adapters=None):
        self._name = name
        self._conda_env_name = "climate_health_" + self._name
        self._setup_command = setup_command
        self._train_command = train_command
        self._predict_command = predict_command
        self._data_type = data_type
        self._conda_env_file = conda_env_file
        self._model = None
        self._working_dir = working_dir
        self._adapters = adapters

    def _run(self, command):
        return run_command(command, working_directory=self._working_dir)

    def run_through_conda(self, command: str):
        if self._conda_env_file:
            return self._run(f'conda run -n {self._conda_env_name} {command}')
        return self._run(command)

    def setup(self):
        if self._conda_env_file:
            try:
                run_command(f'conda env create --name {self._conda_env_name} -f {self._conda_env_file}',
                            self._working_dir)
            except subprocess.CalledProcessError:
                # TODO: This logic is not sound since new entries in env file will not be added to the environment if it exists
                logging.info("Ignoring error when creating conda environment")
                pass

        elif self._setup_command is not None:
            self.run_through_conda(self._setup_command)

    def deactivate(self):
        if self._conda_env_file:
            run_command(f'conda deactivate', self._working_dir)

    def _adapt_data(self, data: pd.DataFrame):
        if self._adapters is None:
            return data
        for to_name, from_name in self._adapters.items():
            if from_name == 'week':
                data[to_name] = data['time_period'].dt.week
            elif from_name == 'month':
                data[to_name] = data['time_period'].dt.month
            elif from_name == 'year':
                data[to_name] = data['time_period'].dt.year
            elif from_name == 'population':
                data[to_name] = 200000  # HACK: This is a placeholder for the population data
                logger.warning("Population data is not available, using placeholder value")
            else:
                data[to_name] = data[from_name]
        return data

    def train(self, train_data: IsSpatioTemporalDataSet[FeatureType]):
        self._model_file_name = self._name + ".model"
        with tempfile.NamedTemporaryFile() as train_datafile:
            train_file_name = train_datafile.name
            with open(train_file_name, "w") as train_datafile:
                pd = train_data.to_pandas()
                new_pd = self._adapt_data(pd)
                new_pd.to_csv(train_file_name)
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
                    raise ValueError(f"No prediction data written to file {out_file.name}")


def run_command(command: str, working_directory="./"):
    """Runs a unix command using subprocess"""
    logging.info(f"Running command: {command}")
    command = command.split()

    try:
        output = subprocess.check_output(command, cwd=working_directory)
        logging.info(output)
    except subprocess.CalledProcessError as e:
        error = e.output.decode()
        logging.info(error)
        raise e

    return output


def get_model_from_yaml_file(yaml_file: str) -> ExternalCommandLineModel:
    # read yaml file into a dict
    with open(yaml_file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    name = data['name']
    train_command = data['train_command']
    predict_command = data['predict_command']
    setup_command = data.get('setup_command', None)
    conda_env_file = data['conda'] if 'conda' in data else None
    data_type = data.get('data_type', None)
    allowed_data_types = {'HealthData': HealthData}
    data_type = allowed_data_types.get(data_type, None)

    model = ExternalCommandLineModel(
        name=name,
        train_command=train_command,
        predict_command=predict_command,
        data_type=data_type,
        setup_command=setup_command,
        conda_env_file=conda_env_file,
        working_dir=Path(yaml_file).parent,
        adapters=data.get('adapters', None)
    )

    return model
