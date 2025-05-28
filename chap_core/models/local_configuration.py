# Module for parsing local configuration of models, i.e. files that are put in config/models directory.
import logging
from pydantic import BaseModel, parse_obj_as

# from pydantic.type_adapter import validate_python
import yaml

from chap_core.database.model_templates_and_config_tables import ModelConfiguration

logger = logging.getLogger(__name__)


class LocalModelTemplateWithConfigurations(BaseModel):
    """Class only used for parsing ModelTemplate from config/models/*.yaml files."""

    url: str
    versions: dict[str, str]
    configurations: dict[str, ModelConfiguration] = {"default": ModelConfiguration()}


Configurations = list[LocalModelTemplateWithConfigurations]


def parse_local_model_config_file(file_name) -> Configurations:
    """
    Reads the local model configuration file and returns a Configurations object.
    The configuration file is in the config/models directory.
    """
    # parse the yaml file using the pydantic model
    with open(file_name, "r") as file:
        content = yaml.safe_load(file)
        configurations = parse_obj_as(
            list[LocalModelTemplateWithConfigurations], content
        )  # change to validate_python in future
        # return
        return configurations


def parse_local_model_config_from_directory(directory, search_pattern="*.yaml") -> Configurations:
    """
    Reads the local model configuration files from the config/models directory and returns a Configurations object.
    The configuration files are in the config/models directory.
    """

    # First look for the default.yaml file, we only read the lastest version from this file
    logger.info("Parsing default model configs")
    default_file = directory / "default.yaml"
    default_configurations = parse_local_model_config_file(default_file)

    # for every model template in default.yaml, keep only the version defined last
    # in the file, and remove all other versions
    for config in default_configurations:
        old_versions = list(config.versions.items())
        new_versions = old_versions[-1:]  # keep only the last version
        config.versions = dict(new_versions)

    all_configurations = default_configurations

    # Now read all the other yaml files in the directory
    for file in directory.glob(search_pattern):
        if file.name == "default.yaml":
            continue
        logger.info(f"Parsing custom model config file {file}")
        file_configurations = parse_local_model_config_file(file)
        for config in file_configurations:
            all_configurations.append(config)

    return all_configurations
