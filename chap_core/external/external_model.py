import logging
import os.path
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeVar

import git
import yaml

from chap_core._legacy_dataset import IsSpatioTemporalDataSet
from chap_core.datatypes import (
    ClimateHealthTimeSeries,
    ClimateData,
    HealthData,
)
from chap_core.external.mlflow import (
    ExternalModel,
    get_train_predict_runner,
)
from chap_core.runners.command_line_runner import CommandLineRunner
from chap_core.runners.docker_runner import DockerImageRunner, DockerRunner
from chap_core.runners.runner import Runner

logger = logging.getLogger(__name__)


class IsExternalModel(Protocol):
    def get_predictions(
        self,
        train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
        future_climate_data: IsSpatioTemporalDataSet[ClimateData],
    ) -> IsSpatioTemporalDataSet[HealthData]: ...


FeatureType = TypeVar("FeatureType")


class SimpleFileContextManager:
    def __init__(self, filename, mode="r"):
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



def get_runner_from_yaml_file(yaml_file: str) -> Runner:
    with open(yaml_file, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        working_dir = Path(yaml_file).parent

        if "dockerfile" in data:
            return DockerImageRunner(data["dockerfile"], working_dir)
        elif "dockername" in data:
            return DockerRunner(data["dockername"], working_dir)
        elif "conda" in data:
            raise Exception("Conda runner not implemented")
        else:
            return CommandLineRunner(working_dir)


def get_model_from_directory_or_github_url(model_path, base_working_dir=Path("runs/"), ignore_env=False,
                                           make_run_dir=True):
    """
    Gets the model and initializes a working directory with the code for the model.
    model_path can be a local directory or github url
    """
    is_github = False
    commit = None
    if isinstance(model_path, str) and model_path.startswith("https://github.com"):
        dir_name = model_path.split("/")[-1].replace(".git", "")
        model_name = dir_name
        if '@' in model_path:
            model_path, commit = model_path.split('@')
        is_github = True
    else:
        model_name = Path(model_path).name

    if not make_run_dir:
        working_dir = base_working_dir / model_name / "latest"
        # clear working dir
        if working_dir.exists():
            logging.info(f"Removing previous working dir {working_dir}")
            shutil.rmtree(working_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        working_dir = base_working_dir / model_name / timestamp
        # check that working dir does not exist
        assert not working_dir.exists(), f"Working dir {working_dir} already exists. This should not happen if make_run_dir is True"

    logger.info(f"Writing results to {working_dir}")

    if is_github:
        working_dir.mkdir(parents=True)
        repo = git.Repo.clone_from(model_path, working_dir)
        if commit:
            logger.info(f'Checking out commit {commit}')
            repo.git.checkout(commit)

    else:
        # copy contents of model_path to working_dir
        logger.info(f"Copying files from {model_path} to {working_dir}")
        shutil.copytree(model_path, working_dir)

    # assert that a config file exists
    if (working_dir / "MLproject").exists():
        assert (working_dir / "MLproject").exists(), f"MLproject file not found in {working_dir}"
        return get_model_from_mlproject_file(working_dir / "MLproject", ignore_env=ignore_env)
    elif (working_dir / "config.yml").exists():
        assert False, "config.yml file not supported anymore"
        #return get_model_from_yaml_file(working_dir / "config.yml", working_dir)
    else:
        raise Exception("No config.yml or MLproject file found in model directory")


def get_model_from_mlproject_file(mlproject_file, ignore_env=False) -> ExternalModel:
    """parses file and returns the model
    Will not use MLflows project setup if docker is specified
    """
    #is_in_docker = os.environ.get("IS_IN_DOCKER", False)
    #if is_in_docker:
    #    ignore_env = True

    with open(mlproject_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if "docker_env" in config:
        runner_type = "docker"
    else:
        runner_type = "mlflow"

    runner = get_train_predict_runner(mlproject_file, runner_type, skip_environment=ignore_env)
    logging.info("Runner is %s", runner)
    logging.info("Will create ExternalMlflowModel")
    name = config["name"]
    adapters = config.get("adapters", None)
    allowed_data_types = {"HealthData": HealthData}
    data_type = allowed_data_types.get(config.get("data_type", None), None)
    return ExternalModel(
        runner,
        name=name,
        adapters=adapters,
        data_type=data_type,
        working_dir=Path(mlproject_file).parent,
    )


def get_model_maybe_yaml(model_name):
    model = get_model_from_directory_or_github_url(model_name)
    return model, model.name
