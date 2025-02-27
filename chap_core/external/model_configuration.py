

from typing import Optional
from pydantic import BaseModel


class DockerEnvConfig(BaseModel):
    image: str


class CommandConfig(BaseModel):
    command: str
    parameters: Optional[dict[str, str]] = None


class EntryPointConfig(BaseModel):
    train: CommandConfig
    predict: CommandConfig


class ModelTemplateConfig(BaseModel):
    name: str
    entry_points: EntryPointConfig
    docker_env: Optional[DockerEnvConfig] = None
    python_env: Optional[str] = None
    required_fields: list[str] = ["rainfall", "mean_temperature"]
    allow_free_additional_continuous_covariates: bool = False 
    adapters: Optional[dict[str, str]] = None


