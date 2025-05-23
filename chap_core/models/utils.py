from datetime import datetime
from typing import TYPE_CHECKING, Literal

import git
from chap_core.datatypes import HealthData
from chap_core.exceptions import InvalidModelException
from chap_core.external.external_model import logger
from chap_core.runners.helper_functions import get_train_predict_runner_from_model_template_config
from chap_core.external.model_configuration import ModelTemplateConfig, ModelTemplateConfigV2
from chap_core.models.model_template import ModelTemplate
import shutil
import uuid
import yaml
import logging
import os
from pathlib import Path

if TYPE_CHECKING:
    from chap_core.models.external_model import ExternalModel


def _get_working_dir(model_path, base_working_dir, run_dir_type, model_name):
    if run_dir_type == "use_existing" and not Path(model_path).exists():
        logging.warning(
            f"Model path {model_path} does not exist. Will create a directory for the run (using the name 'latest')"
        )
        run_dir_type = "latest"

    if run_dir_type == "latest":
        working_dir = base_working_dir / model_name / "latest"
        # clear working dir
        if working_dir.exists():
            logger.info(f"Removing previous working dir {working_dir}")
            shutil.rmtree(working_dir)
    elif run_dir_type == "timestamp":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_identifier = timestamp + "_" + str(uuid.uuid4())[:8]
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S%f")
        working_dir = base_working_dir / model_name / unique_identifier
        # check that working dir does not exist
        assert (
            not working_dir.exists()
        ), f"Working dir {working_dir} already exists. This should not happen if make_run_dir is True"
    elif run_dir_type == "use_existing":
        working_dir = Path(model_path)
    else:
        raise ValueError(f"Invalid run_dir_type: {run_dir_type}")

    return run_dir_type, working_dir


def _get_model_code_base(model_path, base_working_dir, run_dir_type):
    is_github = False
    commit = None
    if isinstance(model_path, str) and model_path.startswith("https://github.com"):
        dir_name = model_path.split("/")[-1].replace(".git", "")
        model_name = dir_name
        if "@" in model_path:
            model_path, commit = model_path.split("@")
        is_github = True
    else:
        model_name = Path(model_path).name

    run_dir_type, working_dir = _get_working_dir(model_path, base_working_dir, run_dir_type, model_name)
    logger.info(f"Writing results to {working_dir}")

    if is_github:
        working_dir.mkdir(parents=True)
        repo = git.Repo.clone_from(model_path, working_dir)
        if commit:
            logger.info(f"Checking out commit {commit}")
            repo.git.checkout(commit)
    elif run_dir_type == "use_existing":
        logging.info("Not copying any model files, using existing directory")
    else:
        # copy contents of model_path to working_dir
        logger.info(f"Copying files from {model_path} to {working_dir}")
        shutil.copytree(
            model_path,
            working_dir,
            ignore=lambda dir, contents: list({".venv", "venv"}.intersection(contents)),
            dirs_exist_ok=True,
        )
    return working_dir


def get_model_template_from_mlproject_file(mlproject_file, ignore_env=False) -> ModelTemplate:
    working_dir = Path(mlproject_file).parent

    with open(mlproject_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config = ModelTemplateConfigV2.model_validate(config)

    model_template = ModelTemplate(config, working_dir, ignore_env)
    return model_template


def get_model_template_from_directory_or_github_url(
    model_template_path, base_working_dir=Path("runs/"), ignore_env=False, run_dir_type="timestamp"
) -> ModelTemplate:
    """
    Note: Preferably use ModelTemplate.from_directory_or_github_url instead of
    using this function directly. This function may be depcrecated in the future.

    Gets the model template and initializes a working directory with the code for the model.
    model_path can be a local directory or github url

    Parameters
    ----------
    model_template_path : str
        Path to the model. Can be a local directory or a github url
    base_working_dir : Path, optional
        Base directory to store the working directory, by default Path("runs/")
    ignore_env : bool, optional
        If True, will ignore the environment specified in the MLproject file, by default False
    run_dir_type : Literal["timestamp", "latest", "use_existing"], optional
        Type of run directory to create, by default "timestamp", which creates a new directory based on current timestamp for the run.
        "latest" will create a new directory based on the model name, but will remove any existing directory with the same name.
        "use_existing" will use the existing directory specified by the model path if that exists. If that does not exist, "latest" will be used.
    """

    logger.info(
        f"Getting model template from {model_template_path}. Ignore env: {ignore_env}. Base working dir: {base_working_dir}. Run dir type: {run_dir_type}"
    )
    working_dir = _get_model_code_base(model_template_path, base_working_dir, run_dir_type)

    logger.info(f"Current directory is {os.getcwd()}, working dir is {working_dir.absolute()}")
    assert os.path.isdir(working_dir), working_dir
    assert os.path.isdir(os.path.abspath(working_dir)), working_dir

    # assert that a config file exists
    if not (working_dir / "MLproject").exists():
        raise InvalidModelException("No MLproject file found in model directory")

    template = get_model_template_from_mlproject_file(working_dir / "MLproject", ignore_env=ignore_env)
    return template


def get_model_from_mlproject_file(mlproject_file, ignore_env=False) -> "ExternalModel":
    """parses file and returns the model
    Will not use MLflows project setup if docker is specified
    """

    with open(mlproject_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    working_dir = Path(mlproject_file).parent
    config = ModelTemplateConfig.model_validate(config)
    runner = get_train_predict_runner_from_model_template_config(config, working_dir, ignore_env)
    # runner = get_train_predict_runner(mlproject_file, runner_type, skip_environment=ignore_env)

    logging.info("Runner is %s", runner)
    logging.info("Will create ExternalMlflowModel")
    name = config.name
    adapters = config.adapters  # config.get("adapters", None)
    # allowed_data_types = {"HealthData": HealthData}
    # data_type = allowed_data_types.get(config.get("data_type", None), None)
    data_type = HealthData
    from chap_core.models import ExternalModel

    return ExternalModel(
        runner,
        name=name,
        adapters=adapters,
        data_type=data_type,
        working_dir=Path(mlproject_file).parent,
    )


def get_model_from_directory_or_github_url(
    model_template_path,
    base_working_dir=Path("runs/"),
    ignore_env=False,
    run_dir_type: Literal["timestamp", "latest", "use_existing"] = "timestamp",
    model_configuration_yaml: str = None,
) -> "ExternalModel":
    """
    NOTE: This function is deprecated, can be removed in the future.

    Gets the model and initializes a working directory with the code for the model.
    model_path can be a local directory or github url

    Parameters
    ----------
    model_template_path : str
        Path to the model. Can be a local directory or a github url
    base_working_dir : Path, optional
        Base directory to store the working directory, by default Path("runs/")
    ignore_env : bool, optional
        If True, will ignore the environment specified in the MLproject file, by default False
    run_dir_type : Literal["timestamp", "latest", "use_existing"], optional
        Type of run directory to create, by default "timestamp", which creates a new directory based on current timestamp for the run.
        "latest" will create a new directory based on the model name, but will remove any existing directory with the same name.
        "use_existing" will use the existing directory specified by the model path if that exists. If that does not exist, "latest" will be used.
    model_configuration_yaml : str, optional
        Path to the model configuration yaml file, by default None. This has to be a yaml that is compatible with the model configuration class given by the ModelTemplate.
    """

    template = get_model_template_from_directory_or_github_url(
        model_template_path, ignore_env=ignore_env, run_dir_type=run_dir_type
    )
    model_configuration = None
    # config_class = template.get_config_class()
    if model_configuration_yaml:
        with open(model_configuration_yaml, "r") as file:
            model_configuration = yaml.load(file, Loader=yaml.FullLoader)
            # model_configuration = config_class.model_validate(model_configuration)

    return template.get_model(model_configuration=model_configuration)
