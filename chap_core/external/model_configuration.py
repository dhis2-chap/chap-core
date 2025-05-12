from typing import Optional, Literal
from pydantic import BaseModel


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


class ModelTemplateSchema(BaseModel, extra="forbid"):  # pydantic-specific config to forbid extra fields):
    name: str
    required_fields: list[str] = ["rainfall", "mean_temperature"]
    allow_free_additional_continuous_covariates: bool = False
    user_options: dict = {}
    model_info: Optional[ModelInfo] = None
    adapters: Optional[dict[str, str]] = None #Depracated


class RunnerConfig(BaseModel, extra="forbid"):  # pydantic-specific config to forbid extra fields):
    entry_points: EntryPointConfig
    docker_env: Optional[DockerEnvConfig] = None
    python_env: Optional[str] = None


class ModelTemplateConfig(ModelTemplateSchema, RunnerConfig):
    adapters: Optional[dict[str, str]] = None
