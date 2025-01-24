import os

import numpy as np
import pytest



@pytest.fixture
def example_zip(data_path):
    return data_path / "sample_dhis_data.zip"


@pytest.fixture
def zip_filepath(data_path):
    return data_path / "sample_chap_app_output.zip"


@pytest.fixture
def zip_filepath_population(data_path):
    return data_path / "sample_dhis_data.zip"




