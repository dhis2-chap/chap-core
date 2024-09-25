import os

import numpy as np
import pytest
from chap_core.api import read_zip_folder, train_on_prediction_data
from chap_core.api import dhis_zip_flow


@pytest.fixture
def example_zip(data_path):
    return data_path / "sample_dhis_data.zip"


def test_read_zip_folder(example_zip):
    data = read_zip_folder(example_zip)
    climate_data = data.climate_data
    for location, climate in climate_data.items():
        assert np.issubdtype(climate.data().mean_temperature.dtype, float)


@pytest.fixture
def zip_filepath(data_path):
    return data_path / "sample_chap_app_output.zip"


@pytest.fixture
def zip_filepath_population(data_path):
    return data_path / "sample_dhis_data.zip"


# @pytest.mark.skip(reason='Not finished implementing')
@pytest.mark.slow
@pytest.mark.skip(reason="Outdated")
def test_dhis_zip_flow(models_path, zip_filepath_population):
    out_json = "output.json"
    # model_name = "ewars_Plus"
    model_name = "HierarchicalModel"
    # model_config_file = models_path / "ewars_Plus" / 'config.yml'
    dhis_zip_flow(zip_filepath_population, out_json, model_name)
    # out_json = Path(out_json)
    assert os.path.exists(out_json)
    # Path(out_json).unlink()


@pytest.mark.skip(reason="Failing on CI")
def test_train_on_prediction_data(data_path):
    data = read_zip_folder(data_path / "sample_dhis_data.zip")
    train_on_prediction_data(
        data,
        model_name="external",
        model_path="https://github.com/knutdrand/external_rmodel_example.git",
    )
