import numpy as np
import pytest
from pydantic_geojson import PointModel
from sqlmodel import Session

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.dataset_tables import ObservationBase
from chap_core.rest_api.data_models import DatasetMakeRequest
from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.analytics import BackTestParams, MakeBacktestWithDataRequest
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
def feature_names():
    return ['mean_temperature', 'rainfall', 'population']

@pytest.fixture
def seen_periods():
    return [f'{year}-{month:02d}' for year in range(2020, 2023) for month in range(1, 13)]

@pytest.fixture
def org_units():
    return ['loc_1', 'loc_2', 'loc_3']

@pytest.fixture
def backtest_params():
    return BackTestParams(
        n_periods=3,
        n_splits=2,
        stride=1
    )

@pytest.fixture
def geojson(org_units):
    return FeatureCollectionModel(
        features=[{'type': 'Feature', 'id': ou, 'properties': {'name': ou}, 'geometry': PointModel(coordinates=[0.0, 0.0])} for ou in org_units])


@pytest.fixture
def dataset_make_request(feature_names, seen_periods, backtest_params, org_units, geojson):
    print([(ou_id, ou, fn, tp, float(ou_id+np.sin(t%12)/2))
        for ou_id, ou in enumerate(org_units)
        for fn in feature_names + ['disease_cases']
        for t, tp in enumerate(seen_periods)
    ])


    observations = [ObservationBase(
        org_unit=ou,
        feature_name=fn,
        period=tp,
        value=float(ou_id+np.sin(t%12)/2))
    for ou_id, ou in enumerate(org_units)
    for fn in feature_names + ['disease_cases']
    for t, tp in enumerate(seen_periods)
    ]

    return DatasetMakeRequest(
        name='testing dataset',
        geojson = geojson,
        provided_data = observations,
        data_to_be_fetched = [],
    )

@pytest.fixture
def create_backtest_with_data_request(dataset_make_request, backtest_params)-> MakeBacktestWithDataRequest:
    return MakeBacktestWithDataRequest(
        model_id='naive_model',
        **(dataset_make_request.model_dump() | backtest_params.model_dump())
    )


