import pytest
from climate_health.api import read_zip_folder


@pytest.fixture
def example_zip(data_path):
    return data_path / "sample_chap_app_output.zip"


def test_read_zip_folder(example_zip):
    data = read_zip_folder(example_zip)
