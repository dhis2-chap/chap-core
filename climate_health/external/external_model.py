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
                 name: str,
                 setup_command: str,
                 train_command: str,
                 predict_command: str,
                 data_type: type[FeatureType],
                 conda_env_file: str=None):
        self._name = name
        self._conda_env_name = None
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
            self._conda_env_name = self._name
            try:
                run_command(f'conda env create -f  {self._conda_env_file}')
            except subprocess.CalledProcessError:
                logging.info("Ignoring error when creating conda environment")
                pass

            #run_command(f'conda init')
            #run_command(f'source ~/.bash_profile')
            #run_command(f'conda activate {self._conda_env_file}')

        self.run_through_conda(self._setup_command)

    def deactivate(self):
        if self._conda_env_file:
            run_command(f'conda deactivate')

    def train(self, train_data: IsSpatioTemporalDataSet[FeatureType]):
        self._out_file = self._name + ".Rdata"
        self._saved_state = train_data

        #with tempfile.NamedTemporaryFile() as train_datafile:
        train_file_name = "tmp_train.csv"
        with open(train_file_name, "w") as train_datafile:
            train_data.to_csv(train_file_name)
            command = self._train_command.format(train_data=train_file_name, model=self._out_file)
            response = self.run_through_conda(command)
            print(response)

    def predict(self, future_data: IsSpatioTemporalDataSet[FeatureType]) -> IsSpatioTemporalDataSet[FeatureType]:
        with tempfile.NamedTemporaryFile() as future_datafile:
            future_data.to_csv(future_datafile.name)
            command = self._predict_command.format(future_data=future_datafile.name, model=self._out_file.name)
            response = self.run_through_conda(command)


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
