import pytest
from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health import ExternalModel
from . import EXAMPLE_DATA_PATH
class MockExternalModel:
    pass


@pytest.fixture
def data_set_filename():
    return EXAMPLE_DATA_PATH / 'data.csv'

@pytest.fixture()
def r_script_file_name() -> ClimateHealthTimeSeries:
    return EXAMPLE_DATA_PATH / 'example_r_script.r'

@pytest.mark.skip
def test_external_model_evaluation(r_script_file_name, data_set_filename):
    external_model = ExternalRModel(r_script, lead_time=Month, adaptors=None)
    results_per_year = []
    our_naive_model = NaivePredictor(lead_time=Month)
    naive_results = []
    for train_data, future_climate_data, future_truth in split_test_train_years():
        predictions = external_model.get_predictions(train_data, future_climate_data)
        results_per_year.append(report(future_truth, predictions))
        naive_predictions = NaivePredictor(lead_time=Month).train(train_data).predict(future_climate_data)
        naive_results.append(naive_predictions)
    result: Dashboard = report_results_against_naive(results_per_year, naive_predictions)