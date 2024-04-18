import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from climate_health.datatypes import HealthPopulationData
from climate_health.services.cache_manager import get_cache
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


@pytest.fixture
def data_path():
    return Path(__file__).parent.parent / 'example_data'


@pytest.fixture
def models_path():
    return Path(__file__).parent.parent / 'external_models'


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
