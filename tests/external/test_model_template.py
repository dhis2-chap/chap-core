import pytest

from chap_core.models.utils import get_model_template_from_mlproject_file
from chap_core.models.model_template import ModelTemplate


@pytest.fixture
def model_template(data_path):
    return get_model_template_from_mlproject_file(data_path / 'debug_model' / 'mlproject.yaml')


@pytest.fixture
def model_config_yaml(data_path):
    return data_path / 'debug_model' / 'model_configuration.yaml'


def test_model_template(model_template: ModelTemplate):
    config_class = model_template.get_config_class()
    assert 'additional_continuous_covariates' in config_class.__annotations__, config_class.__annotations__


def test_get_model_from_model_template_with_user_choices(model_template: ModelTemplate):
    user_choices = model_template.get_config_class()

    # fill out some choices that works with this model_template
    #user_choices.additional_continuous_covariates = ["elevation", "population_density"]

    model = model_template.get_model(user_choices(
        n_lag_periods=3,
        additional_continuous_covariates=["elevation", "population_density"])
    )
    assert model.configuration.additional_continuous_covariates == ["elevation", "population_density"]


def test_get_model_template_config_from_yaml(model_template, model_config_yaml):
    config = model_template.get_model_configuration_from_yaml(model_config_yaml)
    print(config)
    assert config.n_lag_periods == 3

