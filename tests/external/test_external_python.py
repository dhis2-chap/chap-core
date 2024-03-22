import pytest

from climate_health.datatypes import ClimateHealthData, HealthData
from climate_health.external.external_model import ExternalCommandLineModel
from climate_health.external.python_model import ExternalPythonModel
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import delta
from .. import TEST_PATH
from ..data_fixtures import full_data, train_data, future_climate_data


@pytest.fixture()
def python_script_file_name() -> str:
    return str(TEST_PATH / 'mock_predictor_script.py')


def test_external_model_evaluation(python_script_file_name, train_data, future_climate_data):
    #external_model = ExternalPythonModel(python_script_file_name, lead_time=delta.Month, adaptors=None)
    path = python_script_file_name

    train_command = "python " + path + " train {train_data} {model}"
    predict_command = "python " + path + " predict {future_data} {model} {out_file}"

    external_model = ExternalCommandLineModel("test_python_model",
                                              train_command,
                                              predict_command,
                                              setup_command=None,
                                              conda_env_file=None,
                                              data_type=HealthData)
    #predictions = external_model.get_predictions(train_data, future_climate_data)
    external_model.setup()
    external_model.train(train_data)
    predictions = external_model.predict(future_climate_data)

    assert isinstance(predictions, SpatioTemporalDict)
    #assert hasattr(predictions, 'get_data_for_location')
