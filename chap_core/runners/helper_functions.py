import logging
from typing import Literal, Optional
from chap_core.external.model_configuration import ModelTemplateConfig
from chap_core.models.model_template import ModelConfiguration
from chap_core.runners.command_line_runner import CommandLineRunner, CommandLineTrainPredictRunner
from chap_core.runners.docker_runner import DockerRunner, DockerTrainPredictRunner
from chap_core.runners.mlflow_runner import MlFlowTrainPredictRunner
from chap_core.runners.runner import TrainPredictRunner
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


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