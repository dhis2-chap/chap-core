from typing import Optional, Literal
from pydantic import BaseModel
from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData, ModelTemplateInformation


# TODO: rename file to model_template_yaml_spec


class DockerEnvConfig(BaseModel):
    image: str


class CommandConfig(BaseModel):
    command: str
    parameters: Optional[dict[str, str]] = None


class EntryPointConfig(BaseModel):
    train: CommandConfig
    predict: CommandConfig


# TODO: remove after refactor
class UserOption(BaseModel):
    name: str
    type: Literal["string", "integer", "float", "boolean"]
    description: str
    default: Optional[str] = None


# TODO: remove after refactor
class ModelInfo(BaseModel):
    author: str
    description: str
    organization: Optional[str]


# TODO: remove after refactor
class ModelTemplateSchema(BaseModel, extra="forbid"):  # pydantic-specific config to forbid extra fields):
    """
    This is all the information that is needed to show the model template in gui
    """

    name: str
    required_covariates: list[str] = ["rainfall", "mean_temperature", "population"]
    allow_free_additional_continuous_covariates: bool = False
    user_options: dict = {}
    model_info: Optional[ModelInfo] = None


class RunnerConfig(BaseModel, extra="forbid"):  # pydantic-specific config to forbid extra fields):
    """This is all needed to actually run model"""

    entry_points: EntryPointConfig
    docker_env: Optional[DockerEnvConfig] = None
    python_env: Optional[str] = None


class ModelTemplateConfigCommon(ModelTemplateInformation, extra="forbid"):
    meta_data: ModelTemplateMetaData = ModelTemplateMetaData()


# TODO: maybe rename to ModelTemplateYamlConfig
class ModelTemplateConfigV2(ModelTemplateConfigCommon, RunnerConfig, extra="forbid"):
    name: str
    source_url: Optional[str] = None
    adapters: Optional[dict[str, str]] = None


# TODO: remove after refactor
class ModelTemplateConfig(ModelTemplateSchema, RunnerConfig):
    """
    TODO: Try to find a better name that is not confusing
    This is all the information that is listed in mlproject file for a model template
    """

    adapters: Optional[dict[str, str]] = None
