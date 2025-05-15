# Module for parsing local configuration of models, i.e. files that are put in config/models directory.
import logging
from pathlib import Path
from typing import Any
from pydantic import BaseModel, parse_obj_as
#from pydantic.type_adapter import validate_python
import yaml

logger = logging.getLogger(__name__)


class LocalModelTemplateConfigurationEntry(BaseModel):
    """Class only used for parsing ModelTemplate from config/models/*.yaml files."""
    url: str
    versions: dict[str, str]
    configurations: dict[str, dict[Any, Any]]


Configurations = dict[str, LocalModelTemplateConfigurationEntry]

def parse_local_model_config_file(file_name) -> Configurations:
    """
    Reads the local model configuration file and returns a Configurations object.
    The configuration file is in the config/models directory.
    """
    # parse the yaml file using the pydantic model
    with open(file_name, "r") as file:
        content = yaml.safe_load(file)
        configurations = parse_obj_as(dict[str, LocalModelTemplateConfigurationEntry], content)  # change to validate_python in future
        return configurations


def parse_local_model_config_from_directory(directory: Path=Path("models")/"config", search_pattern="*.yaml") -> Configurations:
    """
    Reads the local model configuration files from the config/models directory and returns a Configurations object.
    The configuration files are in the config/models directory.
    """

    # First look for the default.yaml file, we only read the lastest version from this file
    default_file = directory / "default.yaml"
    default_configurations = parse_local_model_config_file(default_file)

    # for every model template in default.yaml, keep only the version defined last
    # in the file, and remove all other versions
    for model_name in default_configurations:
        old_versions = list(default_configurations[model_name].versions.items())
        new_versions = old_versions[-1:]  # keep only the last version
        default_configurations[model_name].versions = dict(new_versions)

    all_configurations = default_configurations

    # Now read all the other yaml files in the directory

    for file in directory.glob(search_pattern):
        if file.name == "default.yaml":
            continue
        file_configurations = parse_local_model_config_file(file)
        for template_name, config in file_configurations.items():
            if template_name in all_configurations:
                logger.warning(f"Duplicate template name {template_name} in {file.name}. "
                               "Overwriting with last found from file {file.name}")
                all_configurations[template_name] = config
                    
    return all_configurations

