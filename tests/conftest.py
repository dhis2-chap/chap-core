import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from chap_core.datatypes import HealthPopulationData
from chap_core.services.cache_manager import get_cache
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from .data_fixtures import *

# ignore showing plots in tests
import matplotlib.pyplot as plt
plt.ion()


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark a test as a slow test.")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_integration = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture
def data_path():
    return Path(__file__).parent.parent / "example_data"


@pytest.fixture
def models_path():
    return Path(__file__).parent.parent / "external_models"


@pytest.fixture
def tests_path():
    return Path(__file__).parent


@pytest.fixture(scope="session", autouse=True)
def use_test_cache():
    os.environ["TEST_ENV"] = "true"
    yield
    del os.environ["TEST_ENV"]
    cache = get_cache()
    cache.close()
    shutil.rmtree(cache.directory, ignore_errors=True)


@pytest.fixture()
def health_population_data(data_path):
    file_name = (data_path / "health_population_data").with_suffix(".csv")
    return DataSet.from_pandas(pd.read_csv(file_name), HealthPopulationData)


@pytest.fixture()
def google_earth_engine():
    from chap_core.google_earth_engine.gee_era5 import GoogleEarthEngine

    try:
        return GoogleEarthEngine()
    except:
        pytest.skip("Google Earth Engine not available")


@pytest.fixture
def request_json(data_path):
    return open(data_path / "v1_api/request.json", "r").read()


@pytest.fixture
def big_request_json():
    filepath = "/home/knut/Data/ch_data/chap_request.json"
    if not os.path.exists(filepath):
        pytest.skip()
    with open(filepath, "r") as f:
        return f.read()
