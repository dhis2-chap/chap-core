import pytest

from climate_health.external.python_model import ExternalPythonModel
from climate_health.time_period import delta
from .. import TEST_PATH
from .data_fixtures import train_data, future_climate_data


@pytest.fixture()
def python_script_file_name() -> str:
    return TEST_PATH / 'mock_predictor_script.py'


@pytest.mark.xfail
def test_external_model_evaluation(python_script_file_name, train_data, future_climate_data):
    external_model = ExternalPythonModel(python_script_file_name, lead_time=delta.Month, adaptors=None)
    predictions = external_model.get_predictions(train_data, future_climate_data)
    assert hasattr(predictions, 'get_data_for_location')
