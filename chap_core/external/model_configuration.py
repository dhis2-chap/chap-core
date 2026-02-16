from pydantic import BaseModel

from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation, ModelTemplateMetaData


class DockerEnvConfig(BaseModel):
    image: str


class CommandConfig(BaseModel):
    command: str
    parameters: dict[str, str] | None = None


class EntryPointConfig(BaseModel):
    train: CommandConfig
    predict: CommandConfig


class RunnerConfig(BaseModel, extra="forbid"):  # pydantic-specific config to forbid extra fields):
    """This is all needed to actually run model"""

    entry_points: EntryPointConfig | None = None
    docker_env: DockerEnvConfig | None = None
    python_env: str | None = None
    uv_env: str | None = None
    renv_env: str | None = None


class ModelTemplateConfigCommon(ModelTemplateInformation, extra="forbid"):
    meta_data: ModelTemplateMetaData = ModelTemplateMetaData()


# TODO: maybe rename to ModelTemplateYamlConfig
class ModelTemplateConfigV2(ModelTemplateConfigCommon, RunnerConfig, extra="forbid"):
    """This is used to parse MLProject files"""

    name: str
    source_url: str | None = None
    adapters: dict[str, str] | None = None
    rest_api_url: str | None = None
    version: str | None = None
