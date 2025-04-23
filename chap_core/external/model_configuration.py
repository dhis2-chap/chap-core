from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict


class DockerEnvConfig(BaseModel):
    image: str


class CommandConfig(BaseModel):
    command: str
    parameters: Optional[dict[str, str]] = None


class EntryPointConfig(BaseModel):
    train: CommandConfig
    predict: CommandConfig


class UserOption(BaseModel):
    name: str
    type: Literal["string", "integer", "float", "boolean"]
    description: str
    default: Optional[str] = None


class ModelInfo(BaseModel):
    author: str
    description: str


class ModelTemplateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # pydantic-specific config to forbid extra fields
    name: str
    entry_points: EntryPointConfig
    docker_env: Optional[DockerEnvConfig] = None
    python_env: Optional[str] = None
    required_fields: list[str] = ["rainfall", "mean_temperature"]
    allow_free_additional_continuous_covariates: bool = False
    adapters: Optional[dict[str, str]] = None
    user_options: list[UserOption] = []
    model_info: Optional[ModelInfo] = None
