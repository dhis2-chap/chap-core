import pytest
import yaml

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


def test_model_spec_from_yaml(models_path):
    model_spec = model_spec_from_yaml(models_path / "ewars_Plus/config.yml")
    assert model_spec.name == "ewars_Plus"
    assert model_spec.parameters == EmptyParameterSpec
    assert model_spec.features == [fs.population, fs.rainfall, fs.mean_temperature]
    assert model_spec.period == PeriodType.week



#@pytest.mark.skip('Need a model to test')
def test_model_spec_from_yaml():
    cls = NaiveEstimator
    model_spec = model_spec_from_model(cls)
    assert model_spec.name == "NaiveEstimator"
    assert model_spec.parameters == EmptyParameterSpec
    assert set(model_spec.features) == set([]) #{fs.population, fs.rainfall, fs.mean_temperature}

@pytest.fixture()
def mlflow_yaml(data_path):
    return open(data_path.parent/'external_models'/'naive_python_model_with_mlproject_file_and_docker'/'MLproject').read()

@pytest.fixture()
def chap_ewars_github_url():
    return 'https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6'


def test_model_spec_from_yaml_string(mlflow_yaml):
    # parse yaml string
    data = yaml.safe_load(mlflow_yaml)
    #data = yaml.parse(mlflow_yaml)

    ModelTemplateConfig.model_validate(data)

def test_model_spec_from_github_url(chap_ewars_github_url):
    config =  ExternalModelTemplate.fetch_config_from_github_url(chap_ewars_github_url)
    assert isinstance(config, ModelTemplateConfigV2)



