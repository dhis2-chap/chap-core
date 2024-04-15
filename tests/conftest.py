import os
import shutil
from pathlib import Path

import pytest

from climate_health.services.cache_manager import get_cache


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
