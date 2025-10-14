from unittest.mock import patch
import pytest

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
