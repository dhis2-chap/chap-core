# Module for parsing local configuration of models, i.e. files that are put in config/models directory.

from typing import Any, Optional
from pydantic import BaseModel, RootModel
import yaml


class LocalModelTemplateConfigurationEntry(BaseModel):
    """Class only used for parsing ModelTemplate from config/models/*.yaml files."""
    url: str
    versions: dict[str, str]
    configurations: dict[str, dict[Any, Any]]


class Configurations(RootModel[dict[str, LocalModelTemplateConfigurationEntry]]):
    pass  # a name to a ModelTemplate


def parse_local_model_config_file(file_name) -> Configurations:
    """
    Reads the local model configuration file and returns a Configurations object.
    The configuration file is in the config/models directory.
    """
    # parse the yaml file using the pydantic model
    with open(file_name, "r") as file:
        content = yaml.safe_load(file)
        configurations = Configurations.model_validate(content)
        return configurations


def parse_local_model_configs() -> Configurations:
    """
    Reads the local model configuration files and returns a Configurations object.
    The configuration files are in the config/models directory.
    """
    import os
    from pathlib import Path

    config_path = Path(os.path.dirname(__file__)) / "config" / "models"
    configurations = Configurations(templates={})
    for file in config_path.glob("*.yaml"):
        with open(file, "r") as f:
            configurations.templates[file.stem] = ModelTemplate.parse_raw(f.read())
    return configurations