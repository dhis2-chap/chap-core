import pytest
from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health import ExternalModel

class MockExternalModel:
    pass


@pytest.fixture
def external_model():
    return MockExternalModel()

@pytest.fixture()
def data_set() -> ClimateHealthTimeSeries:
    return MockClimateHealhtData()

@pytest.mark.skip
def test_external_model_evaluation(r_script_file_name, data_set_file_name):
    external_model = ExternalRModel(r_script, lead_time=Month, adaptors=None)
    results_per_year = []
    for train_data, future_climate_data, future_truth in split_test_train_years():
        predictions = external_model.get_predictions(train_data, future_climate_data)
        results_per_year.append(report(future_truth, predictions))
