import os
import pytest
import yaml
import logging
from chap_core.file_io.file_paths import get_example_data_path
logger = logging.getLogger(__name__)

from chap_core.external.model_configuration import ModelTemplateConfigV2, ModelTemplateConfig
from chap_core.model_spec import (
    ModelSpec,
    model_spec_from_yaml,
    PeriodType,
    EmptyParameterSpec,
    model_spec_from_model,
)
import chap_core.predictor.feature_spec as fs
from chap_core.models.model_template import ExternalModelTemplate
from chap_core.predictor.naive_estimator import NaiveEstimator


# TODO: below commented tests are duplicate and probably outdated, ok to delete? 

# def test_model_spec_from_yaml(models_path):
#     model_spec = model_spec_from_yaml(models_path / "ewars_Plus/config.yml")
#     assert model_spec.name == "ewars_Plus"
#     assert model_spec.parameters == EmptyParameterSpec
#     assert model_spec.features == [fs.population, fs.rainfall, fs.mean_temperature]
#     assert model_spec.period == PeriodType.week

# @pytest.mark.skip('Need a model to test')
# def test_model_spec_from_yaml():
#     cls = NaiveEstimator
#     model_spec = model_spec_from_model(cls)
#     assert model_spec.name == "NaiveEstimator"
#     assert model_spec.parameters == EmptyParameterSpec
#     assert set(model_spec.features) == set([])  # {fs.population, fs.rainfall, fs.mean_temperature}


# @pytest.fixture()
# def naive_yaml_string(data_path):
#     return open(
#         data_path.parent / 'external_models' / 'naive_python_model_with_mlproject_file_and_docker' / 'MLproject').read()


# @pytest.fixture()
# def chap_ewars_github_url():
#     return 'https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6'


def mlflow_paths():
    # NOTE: had to hardcode data folder here since cant use fixture as input for parametrize
    template_folder = get_example_data_path() / 'model_templates'
    names = [
        name for name in os.listdir(template_folder)
        if name.endswith('.yaml')
    ]
    return [os.path.join(template_folder, name) for name in names]


@pytest.mark.parametrize('mlflow_path', mlflow_paths())
def test_parse_configured_model_yaml_file(mlflow_path):
    data = yaml.safe_load(open(mlflow_path).read())
    m = ModelTemplateConfigV2.model_validate(data)
    logger.info(m)
    assert isinstance(m, ModelTemplateConfigV2)


# TODO: delete? probably not necessary to test single models like this, enough with list of mlflow_paths? 

# def test_naive_model_from_yaml_string(naive_yaml_string):
#     data = yaml.safe_load(naive_yaml_string)
#     m = ModelTemplateConfig.model_validate(data)


# @pytest.mark.parametrize('github_url', template_urls)
# def test_model_spec_from_github_url(github_url):
#     config = ExternalModelTemplate.fetch_config_from_github_url(github_url)
#     logger.info(config)
#     assert isinstance(config, ModelTemplateConfigV2)
