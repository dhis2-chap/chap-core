from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
import logging
from chap_core.datatypes import HealthData
from chap_core.external.model_configuration import ModelTemplateConfig, ModelTemplateConfigV2
from chap_core.models.configured_model import ModelConfiguration
from chap_core.models.model_template_interface import ModelTemplateInterface
from chap_core.runners.runner import TrainPredictRunner
from chap_core.external.github import fetch_mlproject_content

if TYPE_CHECKING:
    from chap_core.external.external_model import ExternalModel
    from chap_core.runners.runner import TrainPredictRunner

from pydantic import Field, ValidationError, create_model

logger = logging.getLogger(__name__)


class ModelTemplate:
    """
    Represents a Model Template that can generate concrete models.
    A template defines the choices allowed for a model
    """

    def __init__(self, model_template_config: ModelTemplateConfig, working_dir: str, ignore_env=False):
        self._model_template_config = model_template_config
        self._working_dir = working_dir
        self._ignore_env = ignore_env

    @classmethod
    def from_directory_or_github_url(
        cls, model_template_path, base_working_dir=Path("runs/"), ignore_env=False, run_dir_type="timestamp"
    ) -> "ModelTemplate":
        """
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
        from .utils import get_model_template_from_directory_or_github_url

        return get_model_template_from_directory_or_github_url(
            model_template_path, base_working_dir=base_working_dir, ignore_env=ignore_env, run_dir_type=run_dir_type
        )

    @property
    def name(self):
        return self._model_template_config.name

    def get_train_predict_runner(self) -> TrainPredictRunner:
        pass

    def __str__(self):
        return f"ModelTemplate: {self._model_template_config}"

    def get_config_class(self) -> type[ModelConfiguration]:
        """This will probably not be used"""

        fields = {}
        types = {"string": str, "integer": int, "float": float, "boolean": bool}
        if self._model_template_config.allow_free_additional_continuous_covariates:
            fields["additional_continuous_covariates"] = (list[str], [])
        for name, user_option in self._model_template_config.user_options.items():
            T = types[user_option["type"]]
            if user_option.get("default", None) is not None:
                fields[user_option["title"]] = (T, Field(default=T(user_option["default"])))
            else:
                fields[user_option["title"]] = (T, ...)

        # Note that this actually creates a pydantic class dynamically. For instance, if the
        # template has user_options to select a parameterX with default value 0, the class returned could be:
        # class ModelConfiguration(BaseModel):
        #    parameterX: int= Field(default=0)
        # the advantage of creating a pydantic class is that we can get automatic validation and typing
        # when configuring a model template into a model

        return create_model("ModelConfiguration", **fields)

    def get_model_configuration_from_yaml(self, yaml_file: Path) -> ModelConfiguration:
        with open(yaml_file, "r") as file:
            logger.error(f"Reading yaml file {yaml_file}")
            config = yaml.load(file, Loader=yaml.FullLoader)
            logger.info(config)
            try:
                return self.get_config_class().model_validate(config)
            except ValidationError as e:
                logging.error(config)
                raise e

    def get_default_model(self) -> "ExternalModel":
        return self.get_model()

    def get_model(self, model_configuration: ModelConfiguration = None) -> "ExternalModel":
        """
        Returns a model based on the model configuration. The model configuration is an object of the class
        returned by get_model_class (i.e. specified by the user). If no model configuration is passed, the default
        choices are used.

        Parameters
        ----------
        model_configuration : ModelConfiguration, optional
            The configuration for the model, by default None

        Returns
        -------
        ExternalModel
            The model

        """
        # some choices are handled here, others are simply passedd on to the model
        config_passed_to_model = model_configuration

        # config = ModelTemplateConfig.model_validate(model_configuration)
        from chap_core.runners.helper_functions import get_train_predict_runner_from_model_template_config
        from .external_model import ExternalModel

        runner = get_train_predict_runner_from_model_template_config(
            self._model_template_config, self._working_dir, self._ignore_env, model_configuration
        )

        config = self._model_template_config
        name = config.name
        adapters = config.adapters  # config.get("adapters", None)
        data_type = HealthData

        return ExternalModel(
            runner,
            name=name,
            adapters=adapters,
            data_type=data_type,
            working_dir=self._working_dir,
            configuration=config_passed_to_model,
        )


class ExternalModelTemplate(ModelTemplateInterface):
    """
    This class is instanciated when a model is to be run.
    For parsing mlflow and putting into db/rest-api objects, this class should not be used
    """

    def __init__(self, model_template_config: ModelTemplateConfig, working_dir: str, ignore_env=False):
        self._model_template_config = model_template_config
        self._working_dir = working_dir
        self._ignore_env = ignore_env

    @classmethod
    def fetch_config_from_github_url(cls, github_url) -> ModelTemplateConfigV2:
        content = fetch_mlproject_content(github_url)
        return ModelTemplateConfigV2.model_validate(yaml.safe_load(content) | {"source_url": github_url})

    @property
    def model_template_info(self) -> ModelTemplateConfig:
        return self._model_template_config

    @classmethod
    def from_model_template_config(cls, model_template_config: ModelTemplateConfig, working_dir: str, ignore_env=False):
        return cls(ModelTemplate(model_template_config, working_dir, ignore_env))
