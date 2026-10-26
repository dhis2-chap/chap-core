"""Tests for the shared dataset and model template loading helper used by CLI commands."""

import yaml

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._common import load_dataset_and_template
from chap_core.models.model_template import ModelTemplate


def test_load_dataset_and_template_without_configuration(models_path, data_path):
    dataset, template, configuration = load_dataset_and_template(
        str(models_path / "climate_simulation_model"),
        str(data_path / "laos_subset.csv"),
        None,
        RunConfig(ignore_environment=True),
    )

    assert isinstance(template, ModelTemplate)
    assert len(list(dataset.locations())) > 0
    assert configuration is None


def test_load_dataset_and_template_parses_configuration(models_path, data_path, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"user_option_values": {"seed": 3}}))

    _, _, configuration = load_dataset_and_template(
        str(models_path / "climate_simulation_model"),
        str(data_path / "laos_subset.csv"),
        config_file,
        RunConfig(ignore_environment=True),
    )

    assert configuration is not None
    assert configuration.user_option_values == {"seed": 3}
