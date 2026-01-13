import logging
from typing import Literal, Optional
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.models.model_template import ModelConfiguration
from chap_core.runners.command_line_runner import CommandLineRunner, CommandLineTrainPredictRunner
from chap_core.runners.docker_runner import DockerRunner, DockerTrainPredictRunner
from chap_core.runners.mlflow_runner import MlFlowTrainPredictRunner
from chap_core.runners.runner import TrainPredictRunner
from chap_core.runners.uv_runner import UvRunner, UvTrainPredictRunner
from chap_core.runners.renv_runner import RenvRunner, RenvTrainPredictRunner
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def get_train_predict_runner_from_model_template_config(
    model_template_config: ModelTemplateConfigV2,
    working_dir: Path,
    skip_environment=False,
    model_configuration: Optional["ModelConfiguration"] = None,
) -> TrainPredictRunner:
    """
    Utility function that returns a suitbale runner for a model given a ModelTemplateConfig (which contains information
    about what runner the Template says that its models shold use)
    Returns a TrainPredictRunner (e.g. a MlFlowTrainPredictRunner or a DockerTrainPredictRunner) by parsing
    the config for the template.
    """
    if model_template_config.docker_env is not None:
        runner_type = "docker"
    elif model_template_config.uv_env is not None:
        runner_type = "uv"
    elif model_template_config.renv_env is not None:
        runner_type = "renv"
    elif model_template_config.python_env is not None:
        runner_type = "mlflow"
    else:
        runner_type = ""
        skip_environment = True

    logger.debug(f"skip_environment: {skip_environment}, runner_type: {runner_type}")
    logger.debug(f"Model Configuration: {model_configuration}")
    yaml_filename = "model_configuration_for_run.yaml"
    model_configuration_file = working_dir / yaml_filename
    with open(model_configuration_file, "w") as file:
        model_configuration = model_configuration or {}
        d = model_configuration if isinstance(model_configuration, dict) else model_configuration.model_dump()
        yaml.dump(d, file)

    if skip_environment or runner_type in ("docker", "uv", "renv"):
        # read yaml file into a dict
        train_command = model_template_config.entry_points.train.command  # data["entry_points"]["train"]["command"]
        predict_command = (
            model_template_config.entry_points.predict.command
        )  # data["entry_points"]["predict"]["command"]

        # dump model configuration to a tmp file in working_dir, pass this file to the train and predict command
        # pydantic write to yaml
        # under development
        # if model_configuration is not None:
        #     train_command += f" --model_configuration {model_configuration_file}"
        #     predict_command += f" --model_configuration {model_configuration_file}"
        if skip_environment:
            return CommandLineTrainPredictRunner(
                CommandLineRunner(working_dir),
                train_command,
                predict_command,
                model_configuration_filename=yaml_filename,
            )
        elif runner_type == "uv":
            return UvTrainPredictRunner(
                UvRunner(working_dir),
                train_command,
                predict_command,
                model_configuration_filename=yaml_filename,
            )
        elif runner_type == "renv":
            return RenvTrainPredictRunner(
                RenvRunner(working_dir),
                train_command,
                predict_command,
                model_configuration_filename=yaml_filename,
            )
        else:
            assert model_template_config.docker_env is not None
            logging.debug(f"Docker image is {model_template_config.docker_env.image}")
            command_runner = DockerRunner(model_template_config.docker_env.image, working_dir)
            return DockerTrainPredictRunner(command_runner, train_command, predict_command, yaml_filename)
    else:
        # assert model_configuration is None or model_configuration == {}, "ModelConfiguration (for templates) not supported when runner is mlflow for now"
        assert runner_type == "mlflow"
        return MlFlowTrainPredictRunner(
            working_dir,
            model_configuration_filename=yaml_filename,
            train_params=model_template_config.entry_points.train.parameters.keys(),
        )


def get_train_predict_runner(
    mlproject_file: Path, runner_type: Literal["mlflow", "docker", "uv", "renv"], skip_environment=False
) -> TrainPredictRunner:
    """
    Returns a TrainPredictRunner based on the runner_type.
    If runner_type is "mlflow", returns an MlFlowTrainPredictRunner.
    If runner_type is "docker", the mlproject file is parsed to create a runner
    If runner_type is "uv", creates a UvTrainPredictRunner for uv-managed environments
    If runner_type is "renv", creates a RenvTrainPredictRunner for R/renv-managed environments
    if skip_environment, mlflow and docker is not used, instead returning a TrainPredictRunner that uses the command line
    """
    logger.debug(f"skip_environment: {skip_environment}, runner_type: {runner_type}")
    if skip_environment or runner_type in ("docker", "uv", "renv"):
        working_dir = mlproject_file.parent

        # read yaml file into a dict
        with open(mlproject_file, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        train_command = data["entry_points"]["train"]["command"]
        predict_command = data["entry_points"]["predict"]["command"]

        if skip_environment:
            return CommandLineTrainPredictRunner(CommandLineRunner(working_dir), train_command, predict_command)
        elif runner_type == "uv":
            assert "uv_env" in data, "Runner type is uv, but no uv_env in mlproject file"
            return UvTrainPredictRunner(UvRunner(working_dir), train_command, predict_command)
        elif runner_type == "renv":
            assert "renv_env" in data, "Runner type is renv, but no renv_env in mlproject file"
            return RenvTrainPredictRunner(RenvRunner(working_dir), train_command, predict_command)
        else:
            assert "docker_env" in data, "Runner type is docker, but no docker_env in mlproject file"
            logging.info(f"Docker image is {data['docker_env']['image']}")
            command_runner = DockerRunner(data["docker_env"]["image"], working_dir)
            return DockerTrainPredictRunner(command_runner, train_command, predict_command)
    else:
        assert runner_type == "mlflow"
        return MlFlowTrainPredictRunner(mlproject_file.parent)
