import os
from pathlib import Path

import pytest


@pytest.fixture
def data_path():
    return Path(__file__).parent.parent / 'example_data'


@pytest.fixture(scope="session", autouse=True)
def use_test_cache():
    os.environ['TEST_ENV'] = 'true'
    yield
    del os.environ['TEST_ENV']
