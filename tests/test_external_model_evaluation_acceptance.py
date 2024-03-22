import pandas as pd
import pytest

from climate_health.assessment.dataset_splitting import split_test_train_on_period
from climate_health.assessment.multi_location_evaluator import MultiLocationEvaluator
from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData
from climate_health.file_io.load import load_data_set
from climate_health.reports import HTMLReport
from climate_health.predictor.naive_predictor import NaivePredictor, MultiRegionNaivePredictor
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import Month
from climate_health.time_period.dataclasses import Period
from . import EXAMPLE_DATA_PATH, TMP_DATA_PATH, TEST_PATH
from climate_health.external.python_model import ExternalPythonModel


class MockExternalModel:
    pass


@pytest.fixture
def data_set_filename():
    return EXAMPLE_DATA_PATH / 'data.csv'


@pytest.fixture
def dataset_name():
    return 'hydro_met_subset'


@pytest.fixture()
def r_script_filename() -> str:
    return EXAMPLE_DATA_PATH / 'example_r_script.r'


@pytest.fixture()
def python_script_filename() -> str:
    return TEST_PATH / 'mock_predictor_script.py predict-values '


@pytest.fixture()
def python_model_train_command() -> str:
    return TEST_PATH / 'mock_predictor_script.py train'


@pytest.fixture()
def python_model_predict_command() -> str:
    return TEST_PATH / 'mock_predictor_script.py predict'


@pytest.fixture()
def output_filename(tmp_path) -> str:
    return tmp_path / 'output.html'


# Discussion points:
# Should we index on split-timestamp, first time period, or complete time?
def get_split_points_for_data_set(data_set: IsSpatioTemporalDataSet, max_splits: int) -> list[Period]:
    periods = next(iter(
        data_set.data())).data().time_period  # Uses the time for the first location, assumes it to be the same for all!
    return list(periods)[1::(len(periods)-1) // max_splits]


@pytest.fixture
def load_data_func(data_path):
    def load_data_set(data_set_filename: str) -> IsSpatioTemporalDataSet:
        assert data_set_filename == 'hydro_met_subset'
        file_name = (data_path / data_set_filename).with_suffix('.csv')
        return SpatioTemporalDict.from_pandas(pd.read_csv(file_name), ClimateHealthTimeSeries)

    return load_data_set


class ExternalModelMock:

    def __init__(self, *args, **kwargs):
        pass

    def get_predictions(self, train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries]) -> \
    IsSpatioTemporalDataSet[ClimateData]:
        period = next(iter(future_climate_data.data())).data().time_period[:1]
        new_dict = {loc: HealthData(period, data.data().disease_cases[-1:]) for loc, data in
                    train_data.items()}
        return SpatioTemporalDict(new_dict)


# @pytest.mark.xfail
def test_external_model_evaluation(python_script_filename, dataset_name, output_filename, load_data_func):
    #external_model = ExternalModelMock(python_script_filename, adaptors=None)
    external_model = ExternalPythonModel(python_script_filename, adaptors=None)
    data_set = load_data_func(dataset_name)
    evaluator = MultiLocationEvaluator(model_names=['external_model', 'naive_model'], truth=data_set)
    split_points = get_split_points_for_data_set(data_set, max_splits=5)

    for (train_data, future_truth, future_climate_data) in split_test_train_on_period(data_set, split_points,
                                                                                      future_length=None,
                                                                                      include_future_weather=True):
        predictions = external_model.get_predictions(train_data, future_climate_data)
        evaluator.add_predictions('external_model', predictions)
        naive_predictor = MultiRegionNaivePredictor()
        naive_predictor.train(train_data)
        naive_predictions = naive_predictor.predict(future_climate_data)
        evaluator.add_predictions('naive_model', naive_predictions)

    results = evaluator.get_results()
    report = HTMLReport.from_results(results)
    report.save(output_filename)

# Add test-validation-train split
