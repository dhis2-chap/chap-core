from typing import Optional
from pydantic import BaseModel
from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData, ModelTemplateInformation


class DockerEnvConfig(BaseModel):
    image: str


class CommandConfig(BaseModel):
    command: str
    parameters: Optional[dict[str, str]] = None


class EntryPointConfig(BaseModel):
    train: CommandConfig
    predict: CommandConfig


class RunnerConfig(BaseModel, extra="forbid"):  # pydantic-specific config to forbid extra fields):
    """This is all needed to actually run model"""

    entry_points: Optional[EntryPointConfig] = None
    docker_env: Optional[DockerEnvConfig] = None
    python_env: Optional[str] = None
    uv_env: Optional[str] = None
    renv_env: Optional[str] = None


class ModelTemplateConfigCommon(ModelTemplateInformation, extra="forbid"):
    meta_data: ModelTemplateMetaData = ModelTemplateMetaData()


# TODO: maybe rename to ModelTemplateYamlConfig
class ModelTemplateConfigV2(ModelTemplateConfigCommon, RunnerConfig, extra="forbid"):
    """This is used to parse MLProject files"""

    name: str
    source_url: Optional[str] = None
    adapters: Optional[dict[str, str]] = None
    rest_api_url: Optional[str] = None
    version: Optional[str] = None
