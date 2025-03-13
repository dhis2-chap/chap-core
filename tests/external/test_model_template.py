import pytest

from chap_core.external.external_model import get_model_template_from_mlproject_file
from chap_core.external.mlflow_wrappers import ModelTemplate


@pytest.fixture
def model_template(data_path):
    return get_model_template_from_mlproject_file(data_path / 'debug_model' / 'mlproject.yaml')


def test_model_template(model_template: ModelTemplate):
    config_class = model_template.get_config_class()
    assert 'additional_continous_covariates' in config_class.__annotations__, config_class.__annotations__