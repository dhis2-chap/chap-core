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

    assert parsed.entry_points.train.command != ""
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
