import pytest
from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health import ExternalModel
from climate_health.predictor.naive_predictor import NaivePredictor
from climate_health.time_period import Month
from . import EXAMPLE_DATA_PATH, TMP_DATA_PATH, TEST_PATH
from climate_health.external.python_model import ExternalPythonModel
# from .external.r_model import ExternalRModel


class MockExternalModel:
    pass


@pytest.fixture
def data_set_filename():
    return EXAMPLE_DATA_PATH / 'data.csv'

@pytest.fixture()
def r_script_file_name() -> ClimateHealthTimeSeries:
    return EXAMPLE_DATA_PATH / 'example_r_script.r'

@pytest.fixture()
def python_script_file_name() -> ClimateHealthTimeSeries:
    return TEST_PATH / 'mock_predictor_script.py'

def output_file_name() -> str:
    return TMP_DATA_PATH / 'output.md'

@pytest.mark.skip
def test_external_model_evaluation(python_script_file_name, data_set_filename, output_filename):
    external_model = ExternalPythonModel(python_script_file_name, lead_time=Month, adaptors=None)
    data_set = SpatioTemporalDataSet.from_csv(data_set_filename)
    results_per_year = []
    naive_results = []
    for train_data, future_climate_data, future_truth in split_test_train_years(data_set):
        predictions = external_model.get_predictions(train_data, future_climate_data)
        results_per_year.append(report(future_truth, predictions))
        naive_predictions = NaivePredictor(lead_time=Month).train(train_data).predict(future_climate_data)
        naive_results.append(naive_predictions)
    result: Dashboard = report_results_against_naive(results_per_year, naive_predictions)
    result.save(output_filename)

# Add test-validation-train split
