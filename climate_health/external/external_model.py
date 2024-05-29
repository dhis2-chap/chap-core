import logging
import os.path
import subprocess
import sys
import tempfile
from hashlib import md5
from pathlib import Path
from typing import Protocol, Generic, TypeVar, Tuple, Optional

import docker
import numpy as np
import pandas as pd
import pandas.errors
import yaml
import json

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData, SummaryStatistics
from climate_health.docker_helper_functions import create_docker_image, run_command_through_docker_container
from climate_health.runners.command_line_runner import CommandLineRunner
from climate_health.runners.docker_runner import DockerRunner
from climate_health.runners.runner import Runner
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period.date_util_wrapper import TimeDelta, delta_month

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

    """

    def __init__(self, name: str, train_command: str, predict_command: str, data_type: type[FeatureType],
                 setup_command: str = None,
                 working_dir="./",
                 adapters=None,
                 runner: Runner = None):
        self._is_setup = False
        self._name = name
        self._setup_command = setup_command
        self._train_command = train_command
        self._predict_command = predict_command
        self._data_type = data_type
        self._working_dir = working_dir
        self._model = None
        self._adapters = adapters
        self._model_file_name = self._name + ".model"
        self._runner = runner

    @property
    def name(self):
        return self._name

    def __call__(self):
        return self

    @classmethod
    def from_yaml_file(cls, yaml_file: str, working_dir, dockername: Optional[str] = None) -> "ExternalCommandLineModel":
        # read yaml file into a dict
        with open(yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        name = data['name']
        train_command = data['train_command']
        predict_command = data['predict_command']
        setup_command = data.get('setup_command', None)
        #conda_env_file = data['conda'] if 'conda' in data else None
        data_type = data.get('data_type', None)
        allowed_data_types = {'HealthData': HealthData}
        data_type = allowed_data_types.get(data_type, None)
        runner = get_runner_from_yaml_file(yaml_file, dockername)

        model = cls(
            name=name,
            train_command=train_command,
            predict_command=predict_command,
            data_type=data_type,
            setup_command=setup_command,
            working_dir=working_dir,
            #working_dir=Path(yaml_file).parent,
            adapters=data.get('adapters', None),
            runner=runner,
        )

        return model

    def _run_command(self, command):
        """Wrapper for running command, adds working directory"""
        return run_command(command, working_directory=self._working_dir)

    def run_through_container(self, command: str):
        """Runs the command through either conda, docker or directly depending on the config"""
        return self._runner.run_command(command)

    def setup(self):
        pass

    def deactivate(self):
        if self._conda_env_file:
            self._run_command(f'conda deactivate')

    def _adapt_data(self, data: pd.DataFrame, inverse=False):

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
            elif from_name == 'population':
                data[to_name] = 200_000  # HACK: This is a placeholder for the population data
                logger.warning("Population data is not available, using placeholder value")
            else:
                data[to_name] = data[from_name]
        return data
    @property
    def _dir_name(self):
        return Path('./')


    def train(self, train_data: IsSpatioTemporalDataSet[FeatureType], extra_args=None):
        if extra_args is None:
            extra_args = ''
        #with self._provide_temp_file() as train_datafile:
        train_file_name = 'training_data.csv'
        pd = train_data.to_pandas()
        new_pd = self._adapt_data(pd)
        new_pd.to_csv(Path(self._working_dir) / Path(train_file_name))
        command = self._train_command.format(train_data=train_file_name, model=self._model_file_name, extra_args=extra_args)
        response = self.run_through_container(command)
        print(response)
        self._saved_state = train_data

    def predict(self, future_data: IsSpatioTemporalDataSet[FeatureType]) -> IsSpatioTemporalDataSet[FeatureType]:
        name = 'future_data.csv'
        with open(name, "w") as f:

            df = future_data.to_pandas()
            df['disease_cases'] = np.nan

            new_pd = self._adapt_data(df)
            # if self._is_lagged:
            #    ned_pd = pd.concatenate(self._saved_state, new_pd)
            new_pd.to_csv(Path(self._working_dir) / Path(name))
                # TOOD: combine with training data set for lagged models
        command = self._predict_command.format(future_data=name,
                                               model=self._model_file_name,
                                               out_file='predictions.csv')
        response = self.run_through_container(command)
        print(response)
        try:
            df = pd.read_csv(Path(self._working_dir) / 'predictions.csv')
        # our_df = self._adapt_data(df, inverse=True)

        except pandas.errors.EmptyDataError:
            # todo: Probably deal with this in an other way, throw an exception istead
            logging.warning("No data returned from model (empty file from predictions)")
            raise ValueError(f"No prediction data written to file {out_file.name}")
        result_class = SummaryStatistics if 'quantile_low' in df.columns else HealthData
        return SpatioTemporalDict.from_pandas(df, result_class)

    def forecast(self, future_data: SpatioTemporalDict[FeatureType], n_samples=1000, forecast_delta: TimeDelta= 3*delta_month):
        time_period = next(iter(future_data.data())).data().time_period
        n_periods = forecast_delta // time_period.delta
        future_data = SpatioTemporalDict({key: value.data()[:n_periods] for key, value in future_data.items()})
        return self.predict(future_data)

    def prediction_summary(self, future_data: SpatioTemporalDict[FeatureType], n_samples=1000):
        future_data = SpatioTemporalDict({key: value.data()[:1] for key, value in future_data.items()})
        return self.predict(future_data)

    def _provide_temp_file(self):
        return tempfile.NamedTemporaryFile(dir=self._working_dir, delete=False)

    # def forecast(self, future_data: IsSpatioTemporalDataSet[FeatureType]):
    #    cur_dataset = self._saved_state
    #    for period in relevant_period:
    #        model.predict()


# todo: remove this
def run_command(command: str, working_directory=Path(".")):
    from climate_health.runners.command_line_runner import run_command
    run_command(command, working_directory)



class DryModeExternalCommandLineModel(ExternalCommandLineModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file_creation_folder = './drydata'
        self._file_index = 0
        self._execution_code = ''

    def set_file_creation_folder(self, path):
        self._file_creation_folder = path

    def _run_command(self, command):
        self._execution_code += f'cd {self._working_dir}' + os.linesep
        self._execution_code += command + os.linesep

    def _provide_temp_file(self):
        os.makedirs(self._file_creation_folder, exist_ok=True)
        self._file_index += 1
        return SimpleFileContextManager(f'{self._file_creation_folder}/file{self._file_index}.txt', mode='w+b')

    def get_execution_code(self):
        return self._execution_code


class VerboseRDryModeExternalCommandLineModel(DryModeExternalCommandLineModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._execution_code += f'cd {self._working_dir}' + os.linesep
        self._execution_code += "R" + os.linesep

    def _run_command(self, command):
        command_parts = command.split(" ")
        # assert len(command_parts) == 2, command_parts
        if len(command_parts) > 2:
            r_args = 'c(' + ','.join(['"' + part + '"' for part in command_parts[2:]]) + ')'
            self._execution_code += f'''args = {r_args}''' + os.linesep
        r_lines = open(f'{self._working_dir}/{command_parts[1]}').readlines()
        self._execution_code += os.linesep.join([line for line in r_lines if not 'commandArgs' in line]) + os.linesep


class SimpleFileContextManager:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()

    def write(self, data):
        if self.file:
            self.file.write(data)

    def read(self):
        if self.file:
            return self.file.read()


def get_model_from_yaml_file(yaml_file: str, working_dir, dockername: Optional[str] = None) -> ExternalCommandLineModel:
    return ExternalCommandLineModel.from_yaml_file(yaml_file, working_dir, dockername)


def get_runner_from_yaml_file(yaml_file: str, dockername: Optional[str] = None) -> Runner:
    with open(yaml_file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        working_dir = Path(yaml_file).parent

        if 'dockerfile' in data or dockername is not None:
            if dockername is None:
                dockername = data['dockerfile']
            return DockerRunner(dockername, working_dir)
        elif 'conda' in data:
            raise Exception("Conda runner not implemented")
        else:
            return CommandLineRunner(working_dir)


def get_model_and_runner_from_yaml_file(yaml_file: str) -> Tuple[ExternalCommandLineModel, Runner]:
    return ExternalCommandLineModel.from_yaml_file(yaml_file), get_runner_from_yaml_file(yaml_file)

