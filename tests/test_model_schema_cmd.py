from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from chap_core.cli_endpoints.model import (
    build_example_configuration,
    build_model_configuration_schema,
    schema_cmd,
)
from chap_core.external.model_configuration import ModelTemplateConfigV2


def _make_config(**overrides):
    base = {
        "name": "test_model",
        "user_options": {
            "n_lags": {"type": "integer", "default": 3, "description": "Lags"},
            "precision": {"type": "number"},
        },
        "required_covariates": ["rainfall"],
        "allow_free_additional_continuous_covariates": False,
    }
    base.update(overrides)
    return ModelTemplateConfigV2.model_validate(base)


def test_build_schema_wraps_user_options_and_marks_required():
    schema = build_model_configuration_schema(_make_config())

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    user_values = schema["properties"]["user_option_values"]
    assert user_values["properties"] == {
        "n_lags": {"type": "integer", "default": 3, "description": "Lags"},
        "precision": {"type": "number"},
    }
    assert user_values["additionalProperties"] is False
    assert user_values["required"] == ["precision"]


def test_build_schema_restricts_additional_covariates_when_disallowed():
    schema = build_model_configuration_schema(_make_config())
    extras = schema["properties"]["additional_continuous_covariates"]
    assert extras["type"] == "array"
    assert extras["items"] == {"type": "string"}
    assert extras["maxItems"] == 0


def test_build_schema_allows_free_additional_covariates():
    schema = build_model_configuration_schema(_make_config(allow_free_additional_continuous_covariates=True))
    extras = schema["properties"]["additional_continuous_covariates"]
    assert "maxItems" not in extras
    assert extras["items"] == {"type": "string"}


def test_build_example_uses_defaults_and_null_for_required():
    example = build_example_configuration(_make_config())
    assert example == {
        "user_option_values": {"n_lags": 3, "precision": None},
        "additional_continuous_covariates": [],
    }


def test_schema_cmd_example_flag_writes_example_yaml(tmp_path: Path):
    template = MagicMock()
    template.__enter__.return_value = template
    template.__exit__.return_value = False
    template.model_template_config = _make_config()

    output_file = tmp_path / "example.yaml"
    with patch(
        "chap_core.models.model_template.ModelTemplate.from_directory_or_github_url",
        return_value=template,
    ):
        schema_cmd(model_name="ignored", output_file=output_file, example=True)

    parsed = yaml.safe_load(output_file.read_text())
    assert parsed == {
        "user_option_values": {"n_lags": 3, "precision": None},
        "additional_continuous_covariates": [],
    }


def test_schema_cmd_writes_yaml_to_output_file(tmp_path: Path):
    template = MagicMock()
    template.__enter__.return_value = template
    template.__exit__.return_value = False
    template.model_template_config = _make_config()

    output_file = tmp_path / "schema.yaml"
    with patch(
        "chap_core.models.model_template.ModelTemplate.from_directory_or_github_url",
        return_value=template,
    ):
        schema_cmd(model_name="ignored", output_file=output_file)

    parsed = yaml.safe_load(output_file.read_text())
    assert parsed["title"] == "ModelConfiguration for test_model"
    assert "user_option_values" in parsed["properties"]
    assert "additional_continuous_covariates" in parsed["properties"]
