import pytest
import yaml
from chap_core.external.model_configuration import ModelTemplateConfigV2


@pytest.fixture
def example_specification(data_path):
    return data_path / "Mlproject_example"


@pytest.mark.skip("outdated")
def test_parse_model_template_specification(example_specification):
    with open(example_specification, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    parsed = ModelTemplateConfigV2.model_validate(config)

    assert parsed.entry_points.train.command != ""  # type: ignore[union-attr]
    assert parsed.allow_free_additional_continuous_covariates == True
    print(parsed)


@pytest.fixture
def example_config():
    return """name: Weekly AR RNN Model
python_env: python_env.yml
entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python main.py train {train_data} {model}"
  predict:
    parameters:
      model: str
      historic_data: str
      future_data: str
      out_file: str
    command: "python main.py predict {model} {historic_data} {future_data} {out_file}"
    """


def test_parse_deepar_model_as_model_template(example_config):
    # test that old model config can be parsed as model template
    # parse yaml
    example_config = yaml.load(example_config, Loader=yaml.FullLoader)
    parsed = ModelTemplateConfigV2.model_validate(example_config)


def test_mlproject_min_max_prediction_length_parsed_into_model_template(tmp_path):
    """End-to-end: MLproject file declares min/max_prediction_length at the root,
    and those values must reach ModelTemplate.model_template_config so that
    eval_cmd's dispatch logic (and any other consumer of model_information)
    can read them."""
    from chap_core.models.utils import get_model_template_from_mlproject_file

    mlproject_yaml = """name: bounded_horizon_model
min_prediction_length: 2
max_prediction_length: 6
uv_env: pyproject.toml
entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python main.py train {train_data} {model}"
  predict:
    parameters:
      model: str
      historic_data: str
      future_data: str
      out_file: str
    command: "python main.py predict {model} {historic_data} {future_data} {out_file}"
"""
    mlproject_file = tmp_path / "MLproject"
    mlproject_file.write_text(mlproject_yaml)

    template = get_model_template_from_mlproject_file(mlproject_file)

    assert template.model_template_config.min_prediction_length == 2
    assert template.model_template_config.max_prediction_length == 6


def test_mlproject_max_prediction_length_only_parses_with_min_none(tmp_path):
    """Real models in the wild (e.g. chap-models/Vietnam-dengue-superensemble) declare
    only max_prediction_length and leave min_prediction_length unset. Both halves of
    the contract must be visible on the parsed config: max as the declared int, min as None."""
    from chap_core.models.utils import get_model_template_from_mlproject_file

    mlproject_yaml = """name: vietnam_style_model
max_prediction_length: 1
docker_env:
  image: ghcr.io/dhis2-chap/docker_r_inla:master
entry_points:
  train:
    parameters:
      train_data: path
      model: str
    command: "Rscript train.R {train_data} {model}"
  predict:
    parameters:
      model: str
      historic_data: path
      future_data: path
      out_file: path
    command: "Rscript predict.R {model} {historic_data} {future_data} {out_file}"
"""
    mlproject_file = tmp_path / "MLproject"
    mlproject_file.write_text(mlproject_yaml)

    template = get_model_template_from_mlproject_file(mlproject_file)

    assert template.model_template_config.max_prediction_length == 1
    assert template.model_template_config.min_prediction_length is None
