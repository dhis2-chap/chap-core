# Module for parsing local configuration of models, i.e. files that are put in config/models directory.
import logging
import re

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError, model_validator

from chap_core.database.model_templates_and_config_tables import ModelConfiguration

logger = logging.getLogger(__name__)

# A git version must name one commit, with or without the leading @. Branches, tags and
# short shas can move or be ambiguous, so a label could then silently change code.
FULL_COMMIT_SHA_PATTERN = re.compile(r"@?[0-9a-fA-F]{40}")


class LocalModelTemplateWithConfigurations(BaseModel):
    """Class only used for parsing ModelTemplate from config/models/*.yaml files."""

    url: str
    name: str | None = None  # overrides the template name from the model's own config, e.g. to avoid name clashes
    uses_chapkit: bool = False
    versions: dict[str, str]
    configurations: dict[str, ModelConfiguration] = {"default": ModelConfiguration()}

    @model_validator(mode="after")
    def git_versions_are_full_commit_shas(self):
        # Every label is checked, not only the one that gets seeded, so the file cannot
        # express a moving ref under any label. Chapkit services are versioned by the service.
        if self.uses_chapkit:
            return self
        for label, ref in self.versions.items():
            if not FULL_COMMIT_SHA_PATTERN.fullmatch(ref):
                raise ValueError(
                    f"version {label!r} of {self.url} is {ref!r}, but a version must be a full 40-character "
                    "commit sha, not a branch, tag or short sha"
                )
        return self


class MarketplaceModelSeed(BaseModel):
    """A marketplace model to store at startup, with the template and configurations of its verified stable pin.

    The same registration that ``chap-admin install`` makes, without touching Compose:
    the service itself is run by the deployment, for example through an overlay.
    """

    marketplace: str


Configurations = list[LocalModelTemplateWithConfigurations | MarketplaceModelSeed]


def parse_local_model_config_file(file_name) -> Configurations:
    """
    Reads the local model configuration file and returns a Configurations object.
    The configuration file is in the config/models directory.
    """
    # parse the yaml file using the pydantic model
    with open(file_name) as file:
        content = yaml.safe_load(file)
    try:
        return TypeAdapter(Configurations).validate_python(content)
    except ValidationError as e:
        raise ValueError(f"Invalid model configuration file {file_name}: {e}") from e


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
        if isinstance(config, MarketplaceModelSeed):
            continue
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
