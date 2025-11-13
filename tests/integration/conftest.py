import pytest
from sqlmodel import Session

from chap_core.database.dataset_tables import DataSource
from chap_core.rest_api.data_models import DatasetMakeRequest
from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.analytics import MakeBacktestWithDataRequest, data_sources
from chap_core.rest_api.v1.routers.dependencies import get_database_url, get_session, get_settings
from chap_core.rest_api.worker_functions import WorkerConfig


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
    observations = dataset_observations  # (feature_names, org_units, seen_periods)

    return DatasetMakeRequest(
        name="testing dataset",
        geojson=geojson,
        provided_data=observations,
        data_to_be_fetched=[],
        data_sources=[
            DataSource(covariate=fn, data_element_id=f"de_{i}")
            for i, fn in enumerate(feature_names + ["disease_cases"])
        ],
    )


@pytest.fixture
def dataset_make_request_weekly(
    feature_names, seen_periods_weekly, backtest_params, org_units, geojson, dataset_observations_weekly
):
    observations = dataset_observations_weekly  # (feature_names, org_units, seen_periods_weekly)

    return DatasetMakeRequest(
        name="testing dataset weekly",
        geojson=geojson,
        provided_data=observations,
        data_to_be_fetched=[],
        data_sources=[
            DataSource(covariate=fn, data_element_id=f"de_{i}")
            for i, fn in enumerate(feature_names + ["disease_cases"])
        ],
    )


@pytest.fixture
def create_backtest_with_data_request(dataset_make_request, backtest_params) -> MakeBacktestWithDataRequest:
    return MakeBacktestWithDataRequest(
        model_id="naive_model", **(dataset_make_request.model_dump() | backtest_params.model_dump())
    )


@pytest.fixture
def create_backtest_with_weekly_data_request(
    dataset_make_request_weekly, backtest_params
) -> MakeBacktestWithDataRequest:
    return MakeBacktestWithDataRequest(
        model_id="naive_model", **(dataset_make_request_weekly.model_dump() | backtest_params.model_dump())
    )
