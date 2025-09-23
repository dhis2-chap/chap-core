import pytest
from sqlmodel import Session
from tests.integration.rest_api.db_fixtures import seeded_session
from chap_core.rest_api.data_models import DatasetMakeRequest
from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.analytics import MakeBacktestWithDataRequest
from chap_core.rest_api.v1.routers.dependencies import get_database_url, get_session, get_settings
from chap_core.rest_api.worker_functions import WorkerConfig
from tests.integration.rest_api.data_fixtures import feature_names, seen_periods, org_units, backtest_params, geojson, \
    dataset_observations


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


@pytest.fixture
def dataset_make_request(feature_names, seen_periods, backtest_params, org_units, geojson, dataset_observations):
    observations = dataset_observations  #(feature_names, org_units, seen_periods)

    return DatasetMakeRequest(
        name='testing dataset',
        geojson =geojson,
        provided_data = observations,
        data_to_be_fetched = [],
    )


@pytest.fixture
def create_backtest_with_data_request(dataset_make_request, backtest_params)-> MakeBacktestWithDataRequest:
    return MakeBacktestWithDataRequest(
        model_id='naive_model',
        **(dataset_make_request.model_dump() | backtest_params.model_dump())
    )


