from pathlib import Path
from typing import Generic, TypeVar, Literal
import logging
import pandas as pd
import mlflow
import yaml
from mlflow.utils.process import ShellCommandException

from chap_core.datatypes import HealthData, Samples
from chap_core.exceptions import CommandLineException, ModelConfigurationException, ModelFailedException
from chap_core.exceptions import NoPredictionsError
from chap_core.geometry import Polygons
from chap_core.runners.command_line_runner import CommandLineRunner
from chap_core.runners.docker_runner import DockerRunner
from chap_core.runners.runner import TrainPredictRunner
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import TimePeriod

logger = logging.getLogger(__name__)

FeatureType = TypeVar("FeatureType")


def get_train_predict_runner(mlproject_file: Path, runner_type: Literal["mlflow", "docker"], skip_environment=False) -> TrainPredictRunner:
    """
    Returns a TrainPredictRunner based on the runner_type.
    If runner_type is "mlflow", returns an MlFlowTrainPredictRunner.
    If runner_type is "docker", the mlproject file is parsed to create a runner
    if skip_environment, mlflow and docker is not used, instead returning a TrainPredictRunner that uses the command line
    """
    logger.info(f'skip_environement: {skip_environment}, runner_type: {runner_type}')
    if skip_environment or runner_type == "docker":
        working_dir = mlproject_file.parent

        # read yaml file into a dict
        with open(mlproject_file, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        train_command = data["entry_points"]["train"]["command"]
        predict_command = data["entry_points"]["predict"]["command"]

        if skip_environment:
            return CommandLineTrainPredictRunner(CommandLineRunner(working_dir), train_command, predict_command)
        else:
            assert "docker_env" in data, "Runner type is docker, but no docker_env in mlproject file"

        logging.info(f"Docker image is {data['docker_env']['image']}")
        command_runner = DockerRunner(data["docker_env"]["image"], working_dir)
        return DockerTrainPredictRunner(command_runner, train_command, predict_command)
    else:
        assert runner_type == "mlflow"
        return MlFlowTrainPredictRunner(mlproject_file.parent)


class MlFlowTrainPredictRunner(TrainPredictRunner):
    def __init__(self, model_path):
        self.model_path = model_path

    def train(self, train_file_name, model_file_name, polygons_file_name=None):
        logger.info("Training model using MLflow")
        try:
            return mlflow.projects.run(
                str(self.model_path),
                entry_point="train",
                parameters={
                    "train_data": str(train_file_name),
                    "model": str(model_file_name),
                },
                build_image=True,
            )
        except ShellCommandException as e:
            logger.error(
                "Error running mlflow project, might be due to missing pyenv (See: https://github.com/pyenv/pyenv#installation)"
            )
            raise ModelFailedException(str(e))
        except mlflow.exceptions.ExecutionException as e:
            logger.error(
                "Executation of model failed for some reason. Check the logs for more information"
            )
            raise ModelFailedException(str(e))

    def predict(self, model_file_name, historic_data, future_data, output_file, polygons_file_name=None):
        return mlflow.projects.run(
            str(self.model_path),
            entry_point="predict",
            parameters={
                "historic_data": str(historic_data),
                "future_data": str(future_data),
                "model": str(model_file_name),
                "out_file": str(output_file),
            },
        )



class CommandLineTrainPredictRunner(TrainPredictRunner):
    def __init__(self, runner: CommandLineRunner, train_command: str, predict_command: str):
        self._runner = runner
        self._train_command = train_command
        self._predict_command = predict_command

    def _format_command(self, command, keys):
        try:
            return command.format(**keys)
        except KeyError as e:
            raise ModelConfigurationException(f"Was not able to format command {command}. Does the command contain wrong keys or keys that there is not data for in the dataset?") from e
        
    def _handle_polygons(self, command, keys, polygons_file_name=None):
        # adds polygons to keys if polygons exist. Does some checking with compatibility with command
        if polygons_file_name is not None:
            if "{polygons}" not in command:
                logger.warning(f"Dataset has polygons, but command {command} does not ask for polygons. Will not insert polygons into command.")
            else:
                keys["polygons"] = polygons_file_name
        return keys


    def train(self, train_file_name, model_file_name, polygons_file_name=None):
        keys = {"train_data": train_file_name, "model": model_file_name}
        keys = self._handle_polygons(self._train_command, keys, polygons_file_name)
        command = self._format_command(self._train_command, keys)
        return self._runner.run_command(command)

    def predict(self, model_file_name, historic_data, future_data, output_file, polygons_file_name=None):
        keys = {
            "historic_data": historic_data,
            "future_data": future_data,
            "model": model_file_name,
            "out_file": output_file,
        }
        keys = self._handle_polygons(self._predict_command, keys, polygons_file_name)
        command = self._format_command(self._predict_command, keys)
        return self._runner.run_command(command)


class DockerTrainPredictRunner(CommandLineTrainPredictRunner):
    """This is basically a CommandLineTrainPredictRunner, but with a DockerRunner
    instead of a CommandLineRunner as runner"""
    def __init__(self, runner: DockerRunner, train_command: str, predict_command: str):
        super().__init__(runner, train_command, predict_command)

    def teardown(self):
        self._runner.teardown()


class ExternalModel(Generic[FeatureType]):
    """
    Wrapper around an mlflow model with commands for training and predicting
    """

    def __init__(
        self,
        runner: MlFlowTrainPredictRunner | DockerTrainPredictRunner | CommandLineRunner,
        name: str = None,
        adapters=None,
        working_dir="./",
        data_type=HealthData,
    ):
        self._runner = runner  # MlFlowTrainPredictRunner(model_path)
        # self.model_path = model_path
        self._adapters = adapters
        self._working_dir = working_dir
        self._location_mapping = None
        self._model_file_name = "model"
        self._data_type = data_type
        self._name = name
        self._polygons_file_name = None

    @property
    def name(self):
        return self._name

    def __call__(self):
        return self

    def _write_polygons_to_geojson(self, dataset: DataSet, out_file_name):
        if dataset.polygons is not None:
            logging.info(f"Writing polygons to {out_file_name}")
            Polygons(dataset.polygons).to_file(out_file_name)

    def train(self, train_data: DataSet, extra_args=None):
        if extra_args is None:
            extra_args = ""

        train_file_name = "training_data.csv"
        train_file_name_full = Path(self._working_dir) / Path(train_file_name)
        if train_data.polygons is not None:
            self._polygons_file_name = Path(self._working_dir) / "polygons.geojson"
            self._write_polygons_to_geojson(train_data, self._polygons_file_name)
            logging.info(f"Will pass polygons file {self._polygons_file_name} to train command and predict command")

        pd = train_data.to_pandas()
        new_pd = self._adapt_data(pd)
        new_pd.to_csv(train_file_name_full)

        try:
            self._runner.train(train_file_name, self._model_file_name, 
                               polygons_file_name="polygons.geojson" if self._polygons_file_name is not None else None
                               )
        except CommandLineException as e:
            logger.error("Error training model, command failed")
            raise ModelFailedException(str(e))
        return self

    def _adapt_data(self, data: pd.DataFrame, inverse=False):
        if self._location_mapping is not None:
            data["location"] = data["location"].apply(self._location_mapping.name_to_index)
        if self._adapters is None:
            return data
        adapters = self._adapters
        if inverse:
            adapters = {v: k for k, v in adapters.items()}
            # data['disease_cases'] = data[adapters['disase_cases']]
            return data

        for to_name, from_name in adapters.items():
            # ignore if the column is not present
            if from_name == "disease_cases" and "disease_cases" not in data.columns:
                continue

            if from_name == "week":
                logger.info("Converting time period to week number")
                if hasattr(data["time_period"], "dt"):
                    new_val = data["time_period"].dt.week
                    data[to_name] = new_val
                else:
                    data[to_name] = [int(str(p).split("W")[-1]) for p in data["time_period"]]  # .dt.week

            elif from_name == "month":
                logger.info("Converting time period to month")
                if hasattr(data["time_period"], "dt"):
                    data[to_name] = data["time_period"].dt.month
                else:
                    data[to_name] = [int(str(p).split("-")[-1]) for p in data["time_period"]]
            elif from_name == "year":
                logger.info("Converting time period to year")
                if hasattr(data["time_period"], "dt"):
                    data[to_name] = data["time_period"].dt.year
                else:
                    data[to_name] = [
                        int(str(p).split("W")[0]) for p in data["time_period"]
                    ]  # data['time_period'].dt.year
            else:
                data[to_name] = data[from_name]
        return data

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        logging.info("Running predict")
        future_data_name = Path(self._working_dir) / "future_data.csv"
        historic_data_name = Path(self._working_dir) / "historic_data.csv"
        start_time = future_data.start_timestamp
        logger.info(f"Predicting on dataset from {start_time} to {future_data.end_timestamp}")

        for filename, dataset in [
            (future_data_name, future_data),
            (historic_data_name, historic_data),
        ]:
            with open(filename, "w"):
                adapted_dataset = self._adapt_data(dataset.to_pandas())
                adapted_dataset.to_csv(filename)

        predictions_file = Path(self._working_dir) / "predictions.csv"

        # touch predictions.csv
        with open(predictions_file, "w") as _:
            pass

        try:
            self._runner.predict(
                self._model_file_name,
                "historic_data.csv",
                "future_data.csv",
                "predictions.csv",
                "polygons.geojson" if self._polygons_file_name is not None else None,
            )
        except CommandLineException as e:
            logger.error("Error predicting model, command failed")
            raise ModelFailedException(str(e))

        try:
            df = pd.read_csv(predictions_file)
        except pd.errors.EmptyDataError:
            # todo: Probably deal with this in an other way, throw an exception istead
            logger.warning("No data returned from model (empty file from predictions)")
            raise NoPredictionsError("No prediction data written")

        if self._location_mapping is not None:
            df["location"] = df["location"].apply(self._location_mapping.index_to_name)

        time_periods = [TimePeriod.parse(s) for s in df.time_period.astype(str)]
        mask = [start_time <= time_period.start_timestamp for time_period in time_periods]
        df = df[mask]

        self._runner.teardown()

        return DataSet.from_pandas(df, Samples)


