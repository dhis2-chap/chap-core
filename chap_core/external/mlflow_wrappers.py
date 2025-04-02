from pathlib import Path
from typing import Optional, TypeVar, Literal
import logging

# import mlflow
import yaml
from mlflow.utils.process import ShellCommandException
import mlflow.projects
import mlflow.exceptions
from chap_core.exceptions import ModelConfigurationException, ModelFailedException
from chap_core.external.model_configuration import ModelTemplateConfig
from chap_core.models.model_template import ModelConfiguration
from chap_core.runners.command_line_runner import CommandLineRunner
from chap_core.runners.docker_runner import DockerRunner
from chap_core.runners.runner import TrainPredictRunner

logger = logging.getLogger(__name__)
FeatureType = TypeVar("FeatureType")


def get_train_predict_runner_from_model_template_config(model_template_config: ModelTemplateConfig,
                                                        working_dir: Path,
                                                        skip_environment=False,
                                                        model_configuration: Optional['ModelConfiguration'] = None
                                                        ) -> TrainPredictRunner:
    if model_template_config.docker_env is not None:
        runner_type = "docker"
    elif model_template_config.python_env is not None:
        runner_type = "mlflow"
    else:
        runner_type = ""
        skip_environment = True

    logger.info(f'skip_environement: {skip_environment}, runner_type: {runner_type}')

    if skip_environment or runner_type == "docker":

        # read yaml file into a dict
        train_command = model_template_config.entry_points.train.command  # data["entry_points"]["train"]["command"]
        predict_command = model_template_config.entry_points.predict.command  # data["entry_points"]["predict"]["command"]

        # dump model configuration to a tmp file in working_dir, pass this file to the train and predict command
        # pydantic write to yaml
        # under development
        if model_configuration is not None:
            model_configuration_file = working_dir / "model_configuration_for_run.yaml"
            with open(model_configuration_file, "w") as file:
                yaml.dump(model_configuration.model_dump(), file)
            train_command += f" --model_configuration {model_configuration_file}"
            predict_command += f" --model_configuration {model_configuration_file}"

        if skip_environment:
            return CommandLineTrainPredictRunner(CommandLineRunner(working_dir), train_command, predict_command)
        else:
            assert model_template_config.docker_env is not None

        logging.info(f"Docker image is {model_template_config.docker_env.image}")
        command_runner = DockerRunner(model_template_config.docker_env.image, working_dir)
        return DockerTrainPredictRunner(command_runner, train_command, predict_command)
    else:
        assert model_configuration is None, "ModelConfiguration (for templates) not supported when runner is mlflow for now"
        assert runner_type == "mlflow"
        return MlFlowTrainPredictRunner(working_dir)


def get_train_predict_runner(mlproject_file: Path, runner_type: Literal["mlflow", "docker"],
                             skip_environment=False) -> TrainPredictRunner:
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
            raise ModelConfigurationException(
                f"Was not able to format command {command}. Does the command contain wrong keys or keys that there is not data for in the dataset?") from e

    def _handle_polygons(self, command, keys, polygons_file_name=None):
        # adds polygons to keys if polygons exist. Does some checking with compatibility with command
        if polygons_file_name is not None:
            if "{polygons}" not in command:
                logger.warning(
                    f"Dataset has polygons, but command {command} does not ask for polygons. Will not insert polygons into command.")
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




