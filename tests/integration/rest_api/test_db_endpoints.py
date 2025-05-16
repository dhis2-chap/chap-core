import itertools
import json
import time

import numpy as np
import pytest
from datetime import datetime

from pydantic import ValidationError
from sqlmodel import Session

from chap_core.api_types import EvaluationEntry, PredictionEntry, DataList
from chap_core.database.database import SessionWrapper
from chap_core.database.debug import DebugEntry
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.database.tables import PredictionRead, PredictionInfo, BackTest
from chap_core.rest_api_src.data_models import DatasetMakeRequest, FetchRequest, BackTestFull
from chap_core.rest_api_src.v1.rest_api import app
from fastapi.testclient import TestClient

from chap_core.rest_api_src.v1.routers.analytics import MakePredictionRequest
from chap_core.rest_api_src.v1.routers.crud import DatasetCreate, PredictionCreate
from chap_core.database.dataset_tables import DataSet, DataSetWithObservations, ObservationBase
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
client = TestClient(app)


def test_debug(celery_session_worker):
    response = client.post("/v1/crud/debug")
    assert response.status_code == 200
    assert response.json()['id']


def await_result_id(job_id, timeout=30):
    for _ in range(timeout):
        response = client.get(f"/v1/jobs/{job_id}")
        status = response.json()
        if status == 'SUCCESS':
            return client.get(f"/v1/jobs/{job_id}/database_result").json()['id']
        if status == 'FAILURE':
            assert False, ("Job failed", response.json())
        logger.info(status)
        time.sleep(1)
    assert False, "Timed out"


def await_failure(job_id, timeout=30):
    for _ in range(timeout):
        response = client.get(f"/v1/jobs/{job_id}")
        status = response.json()
        if status == 'SUCCESS':
            assert False, ("Job succeeded", response.json())
        if status == 'FAILURE':
            return
        time.sleep(1)
    assert False, "Timed out"


def test_debug_flow(celery_session_worker, clean_engine, dependency_overrides):
    start_timestamp = time.time()
    response = client.post("/v1/crud/debug")
    assert response.status_code == 200
    job_id = response.json()['id']
    db_id = await_result_id(job_id)
    path = f"/v1/crud/debug/{db_id}"
    response = client.get(path)
    data = DebugEntry.model_validate(response.json())
    assert data.timestamp > start_timestamp


# @pytest.mark.slow
@pytest.mark.parametrize("do_filter", [True, False])
def test_backtest_flow(celery_session_worker, clean_engine, dependency_overrides, weekly_full_data, do_filter):
    with SessionWrapper(clean_engine) as session:
        dataset_id = session.add_dataset('full_data', weekly_full_data, 'polygons', dataset_type='evaluation')
    response = client.post("/v1/crud/backtests",
                           json={"datasetId": dataset_id, "modelId": "naive_model"})
    assert response.status_code == 200, response.json()
    job_id = response.json()['id']
    db_id = await_result_id(job_id)
    response = client.get(f"/v1/crud/backtests/{db_id}")

    # just make sure the datasets are valid
    dataset_response = client.get(f"/v1/crud/datasets")

    assert response.status_code == 200, response.json()
    BackTestFull.model_validate(response.json())
    split_period, org_units = None, []
    if do_filter:
        split_period = '2022W30'
        org_units = ['granada']

    params = {'backtestId': db_id, 'quantiles': [0.1, 0.5, 0.9]}
    if do_filter:
        params |= {'splitPeriod': split_period, 'orgUnits': org_units}

    response = client.get(f'/v1/analytics/evaluation-entry',
                          params=params)

    assert response.status_code == 200, response.json()
    evaluation_entries = response.json()
    params = {} if not do_filter else {'orgUnits': org_units}
    actual_cases = client.get(f'/v1/analytics/actualCases/{db_id}', params=params)
    assert actual_cases.status_code == 200, actual_cases.json()
    actual_cases = DataList.model_validate(actual_cases.json())
    for entry in evaluation_entries:
        assert 'splitPeriod' in entry, f'splitPeriod not in entry: {entry.keys()}'
        entry = EvaluationEntry.model_validate(entry)
        if do_filter:
            assert entry.splitPeriod == split_period, (entry.split_period, split_period)
            assert entry.orgUnit in org_units, (entry.org_unit, org_units)
    if do_filter:
        assert {entry['orgUnit'] for entry in evaluation_entries} == set(org_units), (evaluation_entries, org_units)


def test_add_non_full_dataset(celery_session_worker, clean_engine, dependency_overrides, local_data_path):
    filepath = local_data_path / 'test_data/make_dataset_failing_request.json'

    with open(filepath, 'r') as f:
        data = f.read()
        request = DatasetMakeRequest.model_validate_json(data)
        request.type = 'evaluation'
    print(request)
    _make_dataset(request, wanted_field_names=['rainfall', 'mean_temperature'])


def test_add_dataset_flow(celery_session_worker, dependency_overrides, dataset_create: DatasetCreate):
    data = dataset_create.model_dump_json()
    print(json.dumps(data, indent=2))
    response = client.post("/v1/crud/datasets", data=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()['id'])
    response = client.get(f"/v1/crud/datasets/{db_id}")
    assert response.status_code == 200, response.json()
    ds = DataSetWithObservations.model_validate(response.json())

    assert len(ds.observations) > 0
    print(response.json())
    assert 'orgUnit' in response.json()['observations'][0], response.json()['observations'][0].keys()


def test_list_models(celery_session_worker, dependency_overrides):
    response = client.get("/v1/crud/models")
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0
    assert 'id' in response.json()[0]
    for attr_name in ('displayName', 'id', 'description'):
        '''Check these here to make sure camelCase in response'''
        assert attr_name in response.json()[0], response.json()[0].keys()
    models = [ModelSpecRead.model_validate(m) for m in response.json()]
    assert 'chap_ewars_monthly' in (m.name for m in models)
    ewars_model = next(m for m in models if m.name == 'chap_ewars_monthly')
    assert 'population' in (f.name for f in ewars_model.covariates)
    assert ewars_model.source_url.startswith('https:/')


def test_get_data_sources():
    response = client.get("/v1/analytics/data-sources")
    data = response.json()
    assert response.status_code == 200, data
    assert len(data) == 9, data
    assert next(ds for ds in data if 'rainfall' in ds['supportedFeatures'])['dataset'] == 'era5'


@pytest.fixture
def make_prediction_request(make_dataset_request):
    return MakePredictionRequest(model_id='naive_model',
                                 meta_data={'test': 'test'},
                                 **make_dataset_request.dict())


def test_make_prediction_flow(celery_session_worker, dependency_overrides, make_prediction_request):
    data = make_prediction_request.model_dump_json()
    response = client.post("/v1/analytics/make-prediction",
                           data=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()['id'])
    response = client.get(f"/v1/crud/predictions/{db_id}")
    assert response.status_code == 200, response.json()
    ds = PredictionRead.model_validate(response.json())
    assert ds.meta_data == make_prediction_request.meta_data
    assert len(ds.forecasts) > 0
    assert all(len(f.values) for f in ds.forecasts)


@pytest.fixture()
def make_dataset_request(example_polygons) -> DatasetMakeRequest:
    fetch_request = [FetchRequest(feature_name='mean_temperature',
                                  data_source_name='mean_2m_air_temperature')]
    proivided_features = ['rainfall', 'disease_cases', 'population']
    return create_make_data_request(example_polygons, fetch_request, proivided_features)


def create_make_data_request(example_polygons, fetch_request, proivided_features):
    locations = [f.id for f in example_polygons.features]
    periods = [f'{year}-{month:02d}' for year in range(2020, 2024) for month in range(1, 13)]
    combinations = itertools.product(proivided_features, periods, locations)

    request = DatasetMakeRequest(
        name='testing',
        geojson=example_polygons,
        provided_data=[ObservationBase(feature_name=e, period=p, value=i, org_unit=l) for i, (e, p, l) in
                       enumerate(combinations)],
        data_to_be_fetched=fetch_request)

    return request


@pytest.fixture()
def anonymous_make_dataset_request(data_path):
    with open(data_path / 'anonymous_make_dataset_request.json', 'r') as f:
        return DatasetMakeRequest.model_validate_json(f.read())


def test_make_dataset(celery_session_worker, dependency_overrides, make_dataset_request):
    _make_dataset(make_dataset_request)


def test_make_dataset_failing_with_missing_data(celery_session_worker, dependency_overrides, make_dataset_request):
    first_rainfall_idx = next(
        i for i, o in enumerate(make_dataset_request.provided_data) if o.feature_name == 'rainfall')
    r = make_dataset_request.provided_data.pop(first_rainfall_idx)
    assert r.feature_name == 'rainfall'
    with pytest.raises(AssertionError) as excinfo:
        _make_dataset(make_dataset_request)
    print(excinfo)


def test_make_dataset_anonymous(celery_session_worker, dependency_overrides, anonymous_make_dataset_request):
    _make_dataset(anonymous_make_dataset_request)


def test_backtest_flow_from_request(celery_session_worker,
                                    clean_engine, dependency_overrides,
                                    anonymous_make_dataset_request):
    db_id = _make_dataset(anonymous_make_dataset_request)
    response = client.post("/v1/crud/backtests",
                           json={"datasetId": db_id,
                                 "name": 'testing',
                                 "modelId": "naive_model"})
    assert response.status_code == 200, response.json()
    job_id = response.json()['id']
    db_id = await_result_id(job_id, timeout=120)
    response = client.get(f"/v1/crud/backtests/{db_id}")
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data['name'] == 'testing'
    assert data['created'] is not None


def test_compatible_backtests(clean_engine, dependency_overrides):
    with Session(clean_engine) as session:
        backtest = BackTest(dataset_id=1,
                            model_id='naive_model',
                            org_units=['Oslo', 'Bergen'], split_periods=['202201', '202202'])
        matching = BackTest(dataset_id=1,
                            model_id='chap_auto_ewars',
                            org_units=['Bergen', 'Trondheim'],
                            split_periods=['202202', '202203'])
        non_matching = BackTest(dataset_id=1,
                                model_id='auto_regressive_monthly',
                                org_units=['Trondheim'],
                                split_periods=['202203'])

        session.add(backtest)
        session.add(matching)
        session.add(non_matching)
        session.commit()
        backtest_id = backtest.id
        matching_id = matching.id
    url = f"/v1/analytics/compatible-backtests/{backtest_id}"
    print(url)
    response = client.get(url)
    assert response.status_code == 200, response.json()
    ids = {b['id'] for b in response.json()}
    assert matching_id in ids, (matching_id, ids)
    assert backtest_id not in ids, (backtest_id, ids)
    response = client.get(f"/v1/analytics/backtest-overlap/{backtest_id}/{matching_id}")
    assert response.status_code == 200, response.json()
    assert response.json() == {'orgUnits': ['Bergen'], 'splitPeriods': ['202202']}, response.json()


def _make_dataset(make_dataset_request,
                  wanted_field_names=['rainfall', 'disease_cases', 'population', 'mean_temperature']):
    data = make_dataset_request.model_dump_json()
    response = client.post("/v1/analytics/make-dataset",
                           data=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()['id'])
    dataset_list = client.get("/v1/crud/datasets").json()
    assert len(dataset_list) > 0
    assert db_id in {ds['id'] for ds in dataset_list}
    response = client.get(f"/v1/crud/datasets/{db_id}")
    assert response.status_code == 200, response.json()
    ds = DataSetWithObservations.model_validate(response.json())
    population_entries = [o for o in ds.observations if o.feature_name == 'population']
    assert all((o.value is not None) and np.isfinite(o.value) for o in population_entries), population_entries
    assert ds.created is not None
    field_names = {o.feature_name for o in ds.observations}
    for wfn in wanted_field_names:
        assert wfn in field_names, (field_names, wfn)
    # assert 'rainfall' in field_names
    # assert 'disease_cases' in field_names
    # assert 'population' in field_names
    # assert 'mean_temperature' in field_names
    assert len(field_names) == len(wanted_field_names)
    return db_id


@pytest.mark.skip(reason="Failing because of missing geojson file")
def test_add_csv_dataset(celery_session_worker, dependency_overrides, data_path):
    csv_data = open(data_path / 'nicaragua_weekly_data.csv', 'rb')
    geojson_data = open(data_path / 'nicaragua.json', 'rb')
    response = client.post('/v1/crud/datasets/csvFile', files={"csvFile": csv_data, "geojsonFile": geojson_data})
    assert response.status_code == 200, response.json()


def test_full_prediction_flow(celery_session_worker, dependency_overrides, example_polygons):
    model_list = client.get("/v1/crud/models").json()
    models = [ModelSpecRead.model_validate(m) for m in model_list]
    model = next(m for m in models if m.name == 'naive_model')
    features = [f.name for f in model.covariates]
    fetched_feature, *provided_features = features
    provided_features = provided_features + ['disease_cases']
    data_sources = client.get("/v1/analytics/data-sources").json()
    data_source = next(ds for ds in data_sources if fetched_feature in ds['supportedFeatures'])
    fetch_request = [FetchRequest(feature_name=fetched_feature, data_source_name=data_source['name'])]
    request = create_make_data_request(example_polygons, fetch_request, provided_features)
    request = MakePredictionRequest(model_id=model.name, **request.dict())
    data = request.model_dump_json()
    response = client.post("/v1/analytics/make-prediction",
                           data=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()['id'])
    response = client.get('/v1/crud/predictions')
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0
    print([PredictionInfo.model_validate(entry) for entry in response.json()])
    response = client.get(f"/v1/analytics/prediction-entry/{db_id}", params={'quantiles': [0.1, 0.5, 0.9]})
    assert response.status_code == 200, response.json()
    ds = [PredictionEntry.model_validate(entry) for entry in response.json()]
    assert len(ds) > 0
    assert all(pe.quantile in (0.1, 0.5, 0.9) for pe in ds)


def test_failing_jobs_flow(celery_session_worker, dependency_overrides):
    response = client.post("/v1/debug/trigger-exception")
    assert response.status_code == 200
    job_id = response.json()['id']
    await_failure(job_id)
    response = client.get(f'/v1/jobs/{job_id}')
    assert response.status_code == 200
    assert response.json() == 'FAILURE'
