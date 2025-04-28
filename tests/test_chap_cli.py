from unittest.mock import patch

import pytest

from chap_core.chap_cli import harmonize, evaluate, predict
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


# TODO: None of these tests are run due to missing test files
# need to add the test files to the example_data folder


@pytest.fixture
def harmonize_input_path(data_path):
    path = data_path / "unused.json"
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    return path


@pytest.fixture
def harmonize_input_with_points_path(data_path):
    path = data_path / "harmonize_input_with_points.json"
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    return path


@pytest.fixture
def predict_input_path(data_path):
    path = data_path / "v2.csv"
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    return path


@pytest.fixture
def evaluate_input_path(data_path):
    path = data_path / "v2.csv"
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    return path


def test_harmonize(harmonize_input_path, tmp_path, mocked_gee):
    output_path = tmp_path / "output.csv"
    harmonize(harmonize_input_path, output_path)
    assert output_path.exists()
    DataSet.from_csv(output_path, FullData)


def test_harmonize_with_points(harmonize_input_with_points_path, tmp_path, mocked_gee, point_buffer):
    output_path = tmp_path / "output.csv"
    harmonize(harmonize_input_with_points_path, output_path, point_buffer)
    assert output_path.exists()
    DataSet.from_csv(output_path, FullData)


def test_evaluate(evaluate_input_path, tmp_path):
    output_path = tmp_path / "output.csv"
    model_id = "naive_model"
    evaluate(evaluate_input_path, output_path, model_id)
    assert output_path.exists()


@pytest.mark.parametrize("do_summary", [True, False])
def test_predict(predict_input_path, tmp_path, do_summary):
    output_path = tmp_path / "output.csv"
    model_id = "naive_model"
    predict(predict_input_path, output_path, model_id, do_summary=do_summary)
    assert output_path.exists()


if __name__ == '__main__':
    import pathlib
    file_path = "/mnt/c/Users/karimba/Downloads/chap_request_data_2025-03-31T12_11_15.087Z.json"
    output_path = pathlib.Path(__file__).parent / 'test_outputs'
    test_harmonize_with_points(file_path, output_path, None, 0.1)
