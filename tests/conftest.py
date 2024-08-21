import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from climate_health.datatypes import HealthPopulationData
from climate_health.services.cache_manager import get_cache
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False, help="Run integration tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark a test as an integration test.")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

@pytest.fixture
def data_path():
    return Path(__file__).parent.parent / 'example_data'


@pytest.fixture
def models_path():
    return Path(__file__).parent.parent / 'external_models'

@pytest.fixture
def tests_path():
    return Path(__file__).parent


@pytest.fixture(scope="session", autouse=True)
def use_test_cache():
    os.environ['TEST_ENV'] = 'true'
    yield
    del os.environ['TEST_ENV']
    cache = get_cache()
    cache.close()
    shutil.rmtree(cache.directory, ignore_errors=True)

@pytest.fixture()
def health_population_data(data_path):
    file_name = (data_path / 'health_population_data').with_suffix('.csv')
    return SpatioTemporalDict.from_pandas(pd.read_csv(file_name), HealthPopulationData)


@pytest.fixture()
def google_earth_engine():
    from climate_health.google_earth_engine.gee_era5 import GoogleEarthEngine
    try:
        return GoogleEarthEngine()
    except:
        pytest.skip("Google Earth Engine not available")

@pytest.fixture
def request_json(data_path):
    return open(data_path / 'v1_api/request.json', 'r').read()
