import pytest
from climate_health.api import read_zip_folder
from climate_health.api import dhis_zip_flow


@pytest.fixture
def example_zip(data_path):
    return data_path / "sample_chap_app_output.zip"


def test_read_zip_folder(example_zip):
    data = read_zip_folder(example_zip)


@pytest.fixture
def zip_filepath(data_path):
    return data_path / "sample_chap_app_output.zip"


@pytest.mark.skip(reason='Not finished implementing')
def test_dhis_zip_flow(models_path, zip_filepath):
    out_json = "output.json"
    model_name = "ewars_Plus"
    model_config_file = models_path / "ewars_Plus" / 'config.yml'
    dhis_zip_flow(zip_filepath, out_json, model_config_file)
    assert out_json.exists()
    out_json.unlink()
