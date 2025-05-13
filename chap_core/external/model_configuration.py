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
    '''
    This is all the information that is needed to show the model template in gui
    '''
    name: str
    required_fields: list[str] = ["rainfall", "mean_temperature"]
    allow_free_additional_continuous_covariates: bool = False
    user_options: dict = {}
    model_info: Optional[ModelInfo] = None
    adapters: Optional[dict[str, str]] = None


class RunnerConfig(BaseModel, extra="forbid"):  # pydantic-specific config to forbid extra fields):
    '''This is all needed to actually run model'''
    entry_points: EntryPointConfig
    docker_env: Optional[DockerEnvConfig] = None
    python_env: Optional[str] = None


class ModelTemplateConfig(ModelTemplateSchema, RunnerConfig):
    '''
    TODO: Try to find a better name that is not confusing
    This is all the information that is listed in mlproject file for a model template
    '''
    adapters: Optional[dict[str, str]] = None
