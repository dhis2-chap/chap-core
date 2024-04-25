import pytest
from climate_health.api import dhis_zip_flow

@pytest.fixture
def zip_filepath(data_path):
    return data_path / "sample_chap_app_output.zip"



def test_dhis_zip_flow(zip_filepath):
    out_json = "output.json"
    model_name = "ewars_Plus"
    dhis_zip_flow(zip_filepath, out_json, model_name)
    assert out_json.exists()
    out_json.unlink()
