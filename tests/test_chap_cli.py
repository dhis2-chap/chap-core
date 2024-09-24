import pytest

from climate_health.chap_cli import harmonize, evaluate, predict
from climate_health.datatypes import FullData
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet


@pytest.fixture
def v2_path(data_path):
    path = data_path / "v2.json"
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    return path


@pytest.fixture
def data_filename(data_path):
    path = data_path / "v2.csv"
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    return path


def test_harmonize(v2_path, tmp_path):
    output_path = tmp_path / "output.csv"
    harmonize(v2_path, output_path)
    assert output_path.exists()
    DataSet.from_csv(output_path, FullData)


def test_evaluate(data_filename, tmp_path):
    output_path = tmp_path / "output.csv"
    model_id = "naive_model"
    evaluate(data_filename, output_path, model_id)
    assert output_path.exists()


@pytest.mark.parametrize("do_summary", [True, False])
def test_predict(data_filename, tmp_path, do_summary):
    output_path = tmp_path / "output.csv"
    model_id = "naive_model"
    predict(data_filename, output_path, model_id, do_summary=do_summary)
    assert output_path.exists()