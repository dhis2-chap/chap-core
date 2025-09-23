import datetime

import numpy as np
import pytest
from pydantic_geojson import PointModel

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.dataset_tables import DataSet, Observation, DataSource
from chap_core.rest_api.v1.routers.analytics import BackTestParams


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
def geojson(org_units) -> FeatureCollectionModel:
    return FeatureCollectionModel(
        features=[{'type': 'Feature', 'id': ou, 'properties': {'name': ou}, 'geometry': PointModel(coordinates=[0.0, 0.0])} for ou in org_units])


@pytest.fixture
def dataset_observations(feature_names: list[str], org_units: list[str], seen_periods: list[str]) -> list[
    Observation]:
    observations = [Observation(
        org_unit=ou,
        feature_name=fn,
        period=tp,
        value=float(ou_id + np.sin(t % 12) / 2))
        for ou_id, ou in enumerate(org_units)
        for fn in feature_names + ['disease_cases']
        for t, tp in enumerate(seen_periods)
    ]
    return observations


@pytest.fixture
def dataset(org_units, feature_names, seen_periods, dataset_observations, geojson):
    return DataSet(
        name='testing dataset',
        geojson=geojson.model_dump_json(),
        observations=dataset_observations,
        covariates=feature_names,
        created=datetime.datetime.now(),
        data_sources=[DataSource(covariate=fn, data_element_id=f'de_{i}') for i, fn in enumerate(feature_names)]
    )

@pytest.fixture
def prediction(dataset):
    ...