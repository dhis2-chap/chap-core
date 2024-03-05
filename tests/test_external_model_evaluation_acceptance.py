import pytest

from climate_health.assessment.dataset_splitting import split_test_train_on_period
from climate_health.assessment.multi_location_evaluator import MultiLocationEvaluator
from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.file_io.load import load_data_set
from climate_health.reports import HTMLReport
from climate_health.predictor.naive_predictor import NaivePredictor
from climate_health.time_period import Month
from . import EXAMPLE_DATA_PATH, TMP_DATA_PATH, TEST_PATH
from climate_health.external.python_model import ExternalPythonModel


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
    data_set = load_data_set(data_set_filename)
    evaluator = MultiLocationEvaluator(names=['external_model', 'naive_model'], truth=data_set)
    for (train_data, future_climate_data, future_truth) in split_test_train_on_period(data_set):
        predictions = external_model.get_predictions(train_data, future_climate_data)
        evaluator.add_predictions('external_model', predictions)
        naive_predictions = NaivePredictor(lead_time=Month).train(train_data).predict(future_climate_data)
        evaluator.add_predictions('naive_model', naive_predictions)
    results = evaluator.get_results()
    report = HTMLReport.from_results(results).save(output_filename)
    report.save(output_filename)
# Add test-validation-train split
