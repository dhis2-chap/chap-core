import pytest
from sqlmodel import Session

from chap_core.rest_api_src.v1.rest_api import app
from chap_core.rest_api_src.v1.routers.dependencies import get_session, get_database_url, get_settings
from chap_core.rest_api_src.worker_functions import WorkerConfig


@pytest.fixture
def dependency_overrides(clean_engine):
    def get_test_session():
        with Session(clean_engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_database_url] = lambda: clean_engine.url
    app.dependency_overrides[get_settings] = lambda: WorkerConfig(is_test=True)
    yield
    app.dependency_overrides.clear()
