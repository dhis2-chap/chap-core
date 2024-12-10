import pandas as pd
import pytest

from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.datatypes import ClimateHealthTimeSeries, HealthData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from . import EXAMPLE_DATA_PATH, TEST_PATH


class MockExternalModel:
    pass


@pytest.fixture
def data_set_filename():
    return EXAMPLE_DATA_PATH / "data.csv"


@pytest.fixture
def dataset_name():
    return "hydro_met_subset"


@pytest.fixture()
def r_script_filename() -> str:
    return EXAMPLE_DATA_PATH / "example_r_script.r"


@pytest.fixture()
def python_script_filename() -> str:
    return TEST_PATH / "mock_predictor_script.py predict-values "


@pytest.fixture()
def python_model_train_command() -> str:
    return "python " + str(
        TEST_PATH / "mock_predictor_script.py train {train_data} {model}"
    )


@pytest.fixture()
def python_model_predict_command() -> str:
    return "python " + str(
        TEST_PATH / "mock_predictor_script.py predict {future_data} {model} {out_file}"
    )


@pytest.fixture()
def output_filename(tmp_path) -> str:
    return tmp_path / "output.html"


@pytest.fixture
def load_data_func(data_path):
    def load_data_set(data_set_filename: str) -> DataSet:
        assert data_set_filename == "hydro_met_subset"
        file_name = (data_path / data_set_filename).with_suffix(".csv")
        return DataSet.from_pandas(pd.read_csv(file_name), ClimateHealthTimeSeries)

    return load_data_set

