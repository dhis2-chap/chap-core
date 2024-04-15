import logging
import pytest
logging.basicConfig(level=logging.INFO)
from climate_health.external.external_model import get_model_from_yaml_file
from ..data_fixtures import full_data, train_data, future_climate_data
from climate_health.util import conda_available


@pytest.mark.skipif(not conda_available(), reason='requires conda')
def test_r_model_from_folder(models_path, train_data, future_climate_data):
    yaml = models_path / 'testmodel' / 'config.yml'
    model = get_model_from_yaml_file(yaml)
    model.setup()
    model.train(train_data)
    with pytest.raises(ValueError):
        model.predict(future_climate_data)


def test_python_model_from_folder(models_path, train_data, future_climate_data):
    yaml = models_path / 'naive_python_model' / 'config.yml'
    model = get_model_from_yaml_file(yaml)
    model.train(train_data)
    results = model.predict(future_climate_data)
    assert results is not None


@pytest.mark.parametrize('model_directory', ['naive_python_model', 'testmodel'])
def test_all_external_models_acceptance(model_directory, models_path, train_data, future_climate_data):
    """Only tests that the model can be initiated and that train and predict
    can be called without anything failing"""
    yaml = models_path / model_directory / 'config.yml'
    model = get_model_from_yaml_file(yaml)
    model.train(train_data)
    try:
        results = model.predict(future_climate_data)
        assert results is not None
    except ValueError:
        # This is expected for some models
        pass

