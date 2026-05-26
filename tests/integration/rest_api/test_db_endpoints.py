import itertools
import json
import logging
import time
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from chap_core.api_types import DataList, EvaluationEntry, PredictionEntry
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import DataSet, DataSetWithObservations, ObservationBase
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.database.tables import (
    Backtest,
    BacktestRead,
    Prediction,
    PredictionInfo,
    PredictionRead,
)
from chap_core.rest_api.data_models import (
    BacktestFull,
    ConfiguredModelInfoRead,
    DatasetCreate,
    DatasetMakeRequest,
    FetchRequest,
    MakePredictionRequest,
    ModelConfigurationCreate,
    ModelTemplateRead,
)
from chap_core.rest_api.app import app

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
client = TestClient(app)


def await_result_id(job_id, timeout=30):
    for _ in range(timeout):
        response = client.get(f"/v1/jobs/{job_id}")
        status = response.json()
        if status == "SUCCESS":
            return client.get(f"/v1/jobs/{job_id}/database_result").json()["id"]
        if status == "FAILURE":
            res = client.get(f"/v1/jobs/{job_id}/logs").json()
            assert False, ("Job failed", response.json(), res)

        logger.info(status)
        time.sleep(1)
    assert False, "Timed out"


def test_get_metrics(celery_session_worker, clean_engine, dependency_overrides):
    response = client.get("/v1/visualization/metrics/1")
    assert response.status_code == 200
    assert any(metric["id"] == "crps" for metric in response.json())


def test_get_visualizations(celery_session_worker, clean_engine, dependency_overrides):
    response = client.get("/v1/visualization/metric-plots/1")
    assert response.status_code == 200
    assert any(plot["id"] == "metric_by_horizon_mean" for plot in response.json())


def test_visualization_endpoints_404_on_missing_ids(clean_engine, dependency_overrides):
    """Visualization endpoints used to return 200 with `{"error": ...}` bodies for
    unknown backtest/dataset ids and unknown plot names. Each should now produce a
    proper 4xx response."""
    # unknown backtest
    response = client.get("/v1/visualization/metric-plots/metric_by_horizon/999999/crps")
    assert response.status_code == 404, response.text
    # unknown metric on existing backtest path (the missing-metric case is bad input → 400)
    response = client.get("/v1/visualization/metric-plots/metric_by_horizon/999999/no_such_metric")
    # Backtest is checked first → 404. With a real backtest this would be 400; the
    # other 404 assertion already covers the rewrite, so just assert it is not 200.
    assert response.status_code != 200, response.text
    # unknown plot name on dataset-plots / backtest-plots
    response = client.get("/v1/visualization/dataset-plots/no_such_plot/1")
    assert response.status_code == 404, response.text
    response = client.get("/v1/visualization/backtest-plots/no_such_plot/1")
    assert response.status_code == 404, response.text
    # unknown dataset on a real plot name
    plot_types = client.get("/v1/visualization/dataset-plots/").json()
    if plot_types:
        plot_id = plot_types[0]["id"]
        response = client.get(f"/v1/visualization/dataset-plots/{plot_id}/999999")
        assert response.status_code == 404, response.text


# @pytest.mark.slow
@pytest.mark.parametrize("do_filter", [True, False])
@pytest.mark.slow
@pytest.mark.skip(reason="not in use")
def test_backtest_flow(celery_session_worker, clean_engine, dependency_overrides, weekly_full_data, do_filter):
    with SessionWrapper(clean_engine) as session:
        dataset_id = session.add_dataset("full_data", weekly_full_data, "polygons", dataset_type="evaluation")
    response = client.post("/v1/crud/backtests", json={"datasetId": dataset_id, "modelId": "naive_model"})
    assert response.status_code == 200, response.json()
    job_id = response.json()["id"]
    db_id = await_result_id(job_id)
    response = client.get(f"/v1/crud/backtests/{db_id}")

    # just make sure the datasets are valid
    dataset_response = client.get("/v1/crud/datasets")

    assert response.status_code == 200, response.json()
    BacktestFull.model_validate(response.json())
    split_period, org_units = None, []
    if do_filter:
        split_period = "2022W30"
        org_units = ["granada"]

    params = {"backtestId": db_id, "quantiles": [0.1, 0.5, 0.9]}
    if do_filter:
        params |= {"splitPeriod": split_period, "orgUnits": org_units}

    response = client.get("/v1/analytics/evaluation-entry", params=params)

    assert response.status_code == 200, response.json()
    evaluation_entries = response.json()
    params = {} if not do_filter else {"orgUnits": org_units}
    actual_cases_response = client.get(f"/v1/analytics/actualCases/{db_id}", params=params)
    assert actual_cases_response.status_code == 200, actual_cases_response.json()
    actual_cases = DataList.model_validate(actual_cases_response.json())
    for entry in evaluation_entries:
        assert "splitPeriod" in entry, f"splitPeriod not in entry: {entry.keys()}"
        entry = EvaluationEntry.model_validate(entry)
        if do_filter:
            assert entry.splitPeriod == split_period, (entry.split_period, split_period)
            assert entry.orgUnit in org_units, (entry.org_unit, org_units)
    if do_filter:
        assert {entry["orgUnit"] for entry in evaluation_entries} == set(org_units), (evaluation_entries, org_units)


def test_add_non_full_dataset(celery_session_worker, clean_engine, dependency_overrides, local_data_path):
    filepath = local_data_path / "test_data/make_dataset_failing_request.json"

    with open(filepath, "r") as f:
        data = f.read()
        request = DatasetMakeRequest.model_validate_json(data)
        request.type = "evaluation"
    print(request)
    _make_dataset(request, wanted_field_names=["rainfall", "mean_temperature"])


def test_add_dataset_flow(celery_session_worker, dependency_overrides, dataset_create: DatasetCreate):
    data = dataset_create.model_dump(mode="json")
    print(json.dumps(data, indent=2))
    response = client.post("/v1/crud/datasets", json=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()["id"])
    response = client.get(f"/v1/crud/datasets/{db_id}")
    assert response.status_code == 200, response.json()
    ds = DataSetWithObservations.model_validate(response.json())

    assert len(ds.observations) > 0
    print(response.json())
    assert "orgUnit" in response.json()["observations"][0], response.json()["observations"][0].keys()


def test_list_models_alias_is_gone(clean_engine, dependency_overrides):
    """The /v1/crud/models alias was removed; /configured-models is the only path."""
    response = client.get("/v1/crud/models")
    assert response.status_code == 404, response.text


def test_list_configured_models(celery_session_worker, dependency_overrides):
    response = client.get("/v1/crud/configured-models")
    assert response.status_code == 200, response.json()
    assert isinstance(response.json(), list)
    for m in response.json():
        logger.info(m)
    assert len(response.json()) > 0
    assert "id" in response.json()[0]
    for attr_name in ("displayName", "id", "description", "userOptionValues", "additionalContinuousCovariates"):
        """Check these here to make sure camelCase in response"""
        assert attr_name in response.json()[0], response.json()[0].keys()
    models = [ModelSpecRead.model_validate(m) for m in response.json()]
    assert "chap_ewars_monthly" in (m.name for m in models)
    ewars_model = next(m for m in models if m.name == "chap_ewars_monthly")
    assert "population" in (f.name for f in ewars_model.covariates)
    assert ewars_model.source_url is not None
    assert ewars_model.source_url.startswith("https:/")
    assert isinstance(ewars_model.additional_continuous_covariates, list)


def test_list_model_templates(celery_session_worker, dependency_overrides):
    response = client.get("/v1/crud/model-templates")
    assert response.status_code == 200, response.json()
    assert isinstance(response.json(), list)
    for m in response.json():
        logger.info(m)
    assert len(response.json()) > 0
    assert "id" in response.json()[0]
    for attr_name in ("displayName", "id", "description"):
        """Check these here to make sure camelCase in response"""
        assert attr_name in response.json()[0], response.json()[0].keys()
    models = [ModelTemplateRead.model_validate(m) for m in response.json()]
    assert "chap_ewars_monthly" in (m.name for m in models)
    ewars_model = next(m for m in models if m.name == "chap_ewars_monthly")
    assert "population" in [f for f in ewars_model.required_covariates], ewars_model.required_covariates


def test_get_data_sources():
    response = client.get("/v1/analytics/data-sources")
    data = response.json()
    assert response.status_code == 200, data
    assert len(data) == 9, data
    assert next(ds for ds in data if "rainfall" in ds["supportedFeatures"])["dataset"] == "era5"


@pytest.fixture
def make_prediction_request(make_dataset_request):
    return MakePredictionRequest(
        model_id="naive_model", meta_data={"test": "test"}, **make_dataset_request.model_dump()
    )


def test_make_prediction_flow(celery_session_worker, dependency_overrides, make_prediction_request):
    data = make_prediction_request.model_dump_json()
    response = client.post("/v1/analytics/make-prediction", content=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()["id"])
    response = client.get(f"/v1/crud/predictions/{db_id}")
    assert response.status_code == 200, response.json()
    ds = PredictionRead.model_validate(response.json())
    assert ds.meta_data == make_prediction_request.meta_data
    assert len(ds.forecasts) > 0
    assert all(len(f.values) for f in ds.forecasts)


@pytest.fixture()
def make_dataset_request(example_polygons) -> DatasetMakeRequest:
    pytest.skip("We don it support fetching from gee anymore")
    # fetch_request = [FetchRequest(feature_name="mean_temperature", data_source_name="mean_2m_air_temperature")]
    fetch_request = []  # we don't support fetch request anymore
    proivided_features = ["rainfall", "disease_cases", "population"]
    return create_make_data_request(example_polygons, fetch_request, proivided_features)


def create_make_data_request(example_polygons, fetch_request, proivided_features):
    locations = [f.id for f in example_polygons.features]
    periods = [f"{year}-{month:02d}" for year in range(2020, 2024) for month in range(1, 13)]
    combinations = itertools.product(proivided_features, periods, locations)

    request = DatasetMakeRequest(
        name="testing",
        geojson=example_polygons,
        provided_data=[
            ObservationBase(feature_name=e, period=p, value=i, org_unit=l) for i, (e, p, l) in enumerate(combinations)
        ],
        data_to_be_fetched=fetch_request,
    )

    return request


@pytest.fixture()
def anonymous_make_dataset_request(data_path):
    pytest.skip("We don't support requests with fetching from gee anymore")
    with open(data_path / "anonymous_make_dataset_request.json", "r") as f:
        request = DatasetMakeRequest.model_validate_json(f.read())
        request.data_to_be_fetched = []
        return request


def test_make_dataset(celery_session_worker, dependency_overrides, make_dataset_request):
    _make_dataset(make_dataset_request)


def test_make_dataset_return_rejection_summary(celery_session_worker, dependency_overrides, make_dataset_request):
    first_rainfall_idx = next(
        i for i, o in enumerate(make_dataset_request.provided_data) if o.feature_name == "rainfall"
    )
    r = make_dataset_request.provided_data.pop(first_rainfall_idx)
    assert r.feature_name == "rainfall"
    # with pytest.raises(AssertionError) as excinfo:
    _make_dataset(make_dataset_request, expected_rejections=["LGNjeakKI1q"])
    # print(excinfo)


def test_make_dataset_anonymous(celery_session_worker, dependency_overrides, anonymous_make_dataset_request):
    _make_dataset(anonymous_make_dataset_request)


def test_backtest_flow_from_request(
    celery_session_worker, clean_engine, dependency_overrides, anonymous_make_dataset_request
):
    db_id = _make_dataset(anonymous_make_dataset_request)
    response = client.post("/v1/crud/backtests", json={"datasetId": db_id, "name": "testing", "modelId": "naive_model"})
    assert response.status_code == 200, response.json()
    job_id = response.json()["id"]
    db_id = await_result_id(job_id, timeout=120)
    backtests = client.get("/v1/crud/backtests").json()
    assert len(backtests) > 0
    for backtest in backtests:
        assert "dataset" in backtest, backtest
        assert "configuredModel" in backtest, backtest
        assert backtest["dataset"]["id"] is not None, backtest
        assert backtest["configuredModel"]["name"] is not None, backtest
        assert "modelTemplate" in backtest["configuredModel"], backtest["configuredModel"]
    response = client.get(f"/v1/crud/backtests/{db_id}")
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["name"] == "testing"
    assert data["created"] is not None


def test_compatible_backtests(clean_engine, dependency_overrides):
    with Session(clean_engine) as session:
        dataset = DataSet(name="ds", type="testing", created=datetime.now(), covariates=[])
        session.add(dataset)
        session.commit()

        ds_id = dataset.id
        backtest = Backtest(
            dataset_id=ds_id,
            name="testing",
            model_id="naive_model",
            model_db_id=1,
            org_units=["Oslo", "Bergen"],
            split_periods=["202201", "202202"],
        )
        matching = Backtest(
            dataset_id=ds_id,
            name="testing2",
            model_id="chap_auto_ewars",
            model_db_id=1,
            org_units=["Bergen", "Trondheim"],
            split_periods=["202202", "202203"],
        )
        non_matching = Backtest(
            dataset_id=ds_id,
            name="testing3",
            model_id="auto_regressive_monthly",
            model_db_id=1,
            org_units=["Trondheim"],
            split_periods=["202203"],
        )

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
    ids = {b["id"] for b in response.json()}
    assert matching_id in ids, (matching_id, ids)
    assert backtest_id not in ids, (backtest_id, ids)
    response = client.get(f"/v1/analytics/backtest-overlap/{backtest_id}/{matching_id}")
    assert response.status_code == 200, response.json()
    assert response.json() == {"orgUnits": ["Bergen"], "splitPeriods": ["202202"]}, response.json()


def test_get_backtest_bare_route_returns_info(override_session, seeded_session):
    """GET /v1/crud/backtests/{id} (bare, no /info or /full suffix) should return
    the BacktestRead view rather than 405."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None

    response = client.get(f"/v1/crud/backtests/{backtest.id}")
    assert response.status_code == 200, response.text
    BacktestRead.model_validate(response.json())


def test_get_backtest_bare_route_unknown_id_returns_404(clean_engine, dependency_overrides):
    response = client.get("/v1/crud/backtests/999999")
    assert response.status_code == 404, response.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("nPeriods", 0),
        ("nPeriods", -1),
        ("nSplits", 0),
        ("nSplits", -2),
        ("stride", 0),
        ("stride", -1),
    ],
)
def test_create_backtest_rejects_non_positive_params(field, value, clean_engine, dependency_overrides):
    """n_periods, n_splits and stride must be >= 1; Pydantic validation should
    reject anything <= 0 with 422 before the handler runs."""
    payload = {
        "name": "x",
        "datasetId": 1,
        "modelId": "naive_model",
        "nPeriods": 3,
        "nSplits": 2,
        "stride": 1,
    }
    payload[field] = value
    response = client.post("/v1/analytics/create-backtest", json=payload)
    assert response.status_code == 422, response.text


def test_create_backtest_unknown_dataset_returns_404(clean_engine, dependency_overrides):
    """Both /v1/crud/backtests and /v1/analytics/create-backtest should reject
    bogus dataset ids synchronously rather than queueing a job that fails later."""
    crud_payload = {"name": "bogus", "datasetId": 999999, "modelId": "naive_model"}
    response = client.post("/v1/crud/backtests", json=crud_payload)
    assert response.status_code == 404, response.text

    analytics_payload = {
        "name": "bogus",
        "datasetId": 999999,
        "modelId": "naive_model",
        "nPeriods": 3,
        "nSplits": 2,
        "stride": 1,
    }
    response = client.post("/v1/analytics/create-backtest", json=analytics_payload)
    assert response.status_code == 404, response.text


def test_backtest_overlap_error_message_includes_id(clean_engine, dependency_overrides):
    """The error detail must surface the actual id, not its path position."""
    missing_id1 = 888888
    missing_id2 = 999999
    response = client.get(f"/v1/analytics/backtest-overlap/{missing_id1}/{missing_id2}")
    assert response.status_code == 404, response.text
    assert str(missing_id1) in response.json()["detail"], response.json()


def _prediction_setup_payload(backtest_id: int, name: str = "Saved setup") -> dict:
    return {
        "backtestId": backtest_id,
        "name": name,
        "scheduleCronExpression": "0 6 * * 1",
        "scheduleEnabled": True,
        "quantileTargets": [{"quantile": "median", "dataElementId": "DE_MED"}],
    }


def _create_prediction_setup(backtest_id: int, name: str = "Saved setup"):
    return client.post("/v1/crud/prediction-setups", json=_prediction_setup_payload(backtest_id, name))


def test_create_prediction_setup_happy_path(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    response = _create_prediction_setup(backtest.id, "Setup A")
    assert response.status_code == 200, response.json()
    setup_id = response.json()["id"]

    detail = client.get(f"/v1/crud/prediction-setups/{setup_id}")
    assert detail.status_code == 200, detail.json()
    body = detail.json()
    assert body["name"] == "Setup A"
    assert body["backtestId"] == backtest.id
    assert body["scheduleCronExpression"] == "0 6 * * 1"
    assert body["scheduleEnabled"] is True
    assert body["quantileTargets"] == [{"quantile": "median", "dataElementId": "DE_MED"}]
    assert body["configuredModel"] is not None


def test_create_prediction_setup_snapshots_all_dataset_fields(override_session, seeded_session):
    """Verify every snapshot field is copied from the backtest's dataset over the wire."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    dataset = backtest.dataset

    response = _create_prediction_setup(backtest.id, "Snapshot check")
    assert response.status_code == 200, response.json()
    setup_id = response.json()["id"]

    body = client.get(f"/v1/crud/prediction-setups/{setup_id}").json()
    assert body["startPeriod"] == dataset.first_period
    assert body["orgUnits"] == dataset.org_units
    assert body["periodType"] == dataset.period_type
    expected_sources = [{"covariate": s.covariate, "dataElementId": s.data_element_id} for s in dataset.data_sources]
    assert body["covariateSources"] == expected_sources


def test_list_prediction_setups_returns_empty_when_none_exist(clean_engine, dependency_overrides):
    response = client.get("/v1/crud/prediction-setups")
    assert response.status_code == 200
    assert response.json() == []


def test_create_prediction_setup_missing_backtest_returns_404(clean_engine, dependency_overrides):
    response = client.post(
        "/v1/crud/prediction-setups",
        json=_prediction_setup_payload(99999, "ghost"),
    )
    assert response.status_code == 404


def test_create_prediction_setup_invalid_cron_returns_422(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    payload = _prediction_setup_payload(backtest.id, "Bad cron")
    payload["scheduleCronExpression"] = "not a cron expression"
    response = client.post("/v1/crud/prediction-setups", json=payload)
    assert response.status_code == 422


def test_create_prediction_setup_enabled_without_expression_returns_422(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    payload = _prediction_setup_payload(backtest.id, "Enabled but empty")
    payload["scheduleCronExpression"] = None
    payload["scheduleEnabled"] = True
    response = client.post("/v1/crud/prediction-setups", json=payload)
    assert response.status_code == 422


def test_create_prediction_setup_duplicate_returns_409(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    first = _create_prediction_setup(backtest.id, "First")
    assert first.status_code == 200, first.json()
    second = _create_prediction_setup(backtest.id, "Second")
    assert second.status_code == 409


def test_list_prediction_setups_returns_created_setup(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Listed setup")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.get("/v1/crud/prediction-setups")
    assert response.status_code == 200
    items = response.json()
    assert [item["id"] for item in items] == [setup_id]
    assert "predictions" not in items[0]


def test_get_prediction_setup_includes_linked_predictions(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "With predictions")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None
    prediction.prediction_setup_id = setup_id
    seeded_session.add(prediction)
    seeded_session.commit()

    response = client.get(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 200, response.json()
    body = response.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["id"] == prediction.id


def test_get_prediction_setup_not_found_returns_404(clean_engine, dependency_overrides):
    response = client.get("/v1/crud/prediction-setups/99999")
    assert response.status_code == 404


def test_patch_prediction_setup_updates_name_only(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Original")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.patch(f"/v1/crud/prediction-setups/{setup_id}", json={"name": "Renamed"})
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["name"] == "Renamed"
    assert body["scheduleCronExpression"] == "0 6 * * 1"


def test_patch_prediction_setup_updates_multiple_fields_at_once(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Multi-field")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.patch(
        f"/v1/crud/prediction-setups/{setup_id}",
        json={
            "name": "Multi-field renamed",
            "scheduleCronExpression": "*/5 * * * *",
            "scheduleEnabled": True,
            "quantileTargets": [
                {"quantile": "p25", "dataElementId": "DE_P25"},
                {"quantile": "p75", "dataElementId": "DE_P75"},
            ],
        },
    )
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["name"] == "Multi-field renamed"
    assert body["scheduleCronExpression"] == "*/5 * * * *"
    assert body["scheduleEnabled"] is True
    assert body["quantileTargets"] == [
        {"quantile": "p25", "dataElementId": "DE_P25"},
        {"quantile": "p75", "dataElementId": "DE_P75"},
    ]


def test_patch_prediction_setup_can_clear_schedule(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Clearable")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.patch(
        f"/v1/crud/prediction-setups/{setup_id}",
        json={"scheduleCronExpression": None, "scheduleEnabled": False},
    )
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["scheduleCronExpression"] is None
    assert body["scheduleEnabled"] is False


def test_patch_prediction_setup_rejects_immutable_field(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Immutable")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.patch(f"/v1/crud/prediction-setups/{setup_id}", json={"backtestId": 999})
    assert response.status_code == 422


class _NoopRedis:
    """Empty job-meta store: DELETE flow finds no in-flight jobs to cancel."""

    def keys(self, _pattern):
        return []

    def hgetall(self, _key):
        return {}

    def delete(self, _key):
        return 0


def test_delete_prediction_setup_removes_setup_and_keeps_predictions(override_session, seeded_session, monkeypatch):
    from chap_core.rest_api.v1.routers import crud

    monkeypatch.setattr(crud, "redis", _NoopRedis())

    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Deletable")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None
    prediction.prediction_setup_id = setup_id
    seeded_session.add(prediction)
    seeded_session.commit()
    prediction_id = prediction.id

    response = client.delete(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 200, response.json()
    assert response.json() == {"message": "deleted"}

    response = client.get(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 404

    seeded_session.expire_all()
    surviving = seeded_session.exec(select(Prediction).where(Prediction.id == prediction_id)).one()
    assert surviving.prediction_setup_id is None


def test_delete_prediction_setup_without_predictions(override_session, seeded_session, monkeypatch):
    from chap_core.rest_api.v1.routers import crud

    monkeypatch.setattr(crud, "redis", _NoopRedis())

    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Empty setup")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.delete(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 200, response.json()
    assert response.json() == {"message": "deleted"}

    response = client.get(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 404


def test_delete_prediction_setup_not_found_returns_404(clean_engine, dependency_overrides):
    response = client.delete("/v1/crud/prediction-setups/99999")
    assert response.status_code == 404


def test_delete_prediction_setup_sweeps_matching_job_meta_in_redis(override_session, seeded_session, monkeypatch):
    """When deleting a setup, the router should sweep Redis job_meta:* keys for any
    that carry the setup's id and clear them (cancelling in-flight jobs in the process)."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Sweep target")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    class _FakeRedis:
        def __init__(self):
            self.meta = {
                "job_meta:job-1": {"prediction_setup_id": str(setup_id), "status": "SUCCESS"},
                "job_meta:job-2": {"prediction_setup_id": "999", "status": "SUCCESS"},
            }
            self.deleted: list[str] = []

        def keys(self, _pattern):
            return list(self.meta)

        def hgetall(self, key):
            return self.meta[key]

        def delete(self, key):
            self.deleted.append(key)
            self.meta.pop(key, None)
            return 1

    fake_redis = _FakeRedis()
    from chap_core.rest_api.v1.routers import crud

    monkeypatch.setattr(crud, "redis", fake_redis)

    response = client.delete(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 200, response.json()

    assert fake_redis.deleted == ["job_meta:job-1"]
    assert "job_meta:job-2" in fake_redis.meta


def test_backtest_info_exposes_prediction_setup_id_when_setup_exists(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None

    response = client.get(f"/v1/crud/backtests/{backtest.id}/info")
    assert response.status_code == 200, response.json()
    assert response.json()["predictionSetupId"] is None

    created = _create_prediction_setup(backtest.id, "Linked setup")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.get(f"/v1/crud/backtests/{backtest.id}/info")
    assert response.status_code == 200, response.json()
    assert response.json()["predictionSetupId"] == setup_id


def test_backtest_list_exposes_prediction_setup_id(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Listed-link setup")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.get("/v1/crud/backtests")
    assert response.status_code == 200, response.json()
    matching = next(item for item in response.json() if item["id"] == backtest.id)
    assert matching["predictionSetupId"] == setup_id


def test_run_prediction_setup_not_found_returns_404(clean_engine, dependency_overrides, example_polygons):
    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    # /run body doesn't accept dataToBeFetched / dataSources — strip them.
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload["nPeriods"] = 3
    response = client.post("/v1/crud/prediction-setups/99999/run", json=payload)
    assert response.status_code == 404, response.json()


def test_run_prediction_setup_rejects_legacy_fields(override_session, seeded_session, example_polygons):
    """The new /run body must reject legacy fields (dataSources, dataToBeFetched,
    configuredModelWithDataSourceId) so old clients fail loud."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Legacy reject")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    # Body still contains dataToBeFetched + dataSources from DatasetMakeRequest — exactly the legacy fields.
    payload["nPeriods"] = 3
    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("n_periods", [0, -1])
def test_run_prediction_setup_non_positive_n_periods_returns_422(
    override_session, seeded_session, example_polygons, n_periods
):
    """Pydantic gt=0 on RunPredictionSetupRequest.n_periods should reject 0 and negatives
    before the handler runs. Lock in the contract."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, f"n_periods={n_periods}")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload["nPeriods"] = n_periods

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 422, response.json()


def test_run_prediction_setup_empty_provided_data_returns_422(override_session, seeded_session, example_polygons):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Empty data")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    payload = {
        "name": "empty",
        "geojson": example_polygons.model_dump(mode="json"),
        "providedData": [],
        "nPeriods": 3,
    }
    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 422, response.json()


def test_run_prediction_setup_archived_model_returns_409(override_session, seeded_session, example_polygons):
    from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB

    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    configured_model = seeded_session.get(ConfiguredModelDB, backtest.model_db_id)
    assert configured_model is not None
    created = _create_prediction_setup(backtest.id, "Archived target")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    configured_model.archived = True
    seeded_session.add(configured_model)
    seeded_session.commit()

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload["nPeriods"] = 3

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 409, response.json()


def test_run_prediction_setup_logs_rejection_count(
    override_session, seeded_session, example_polygons, monkeypatch, caplog
):
    """When validate_full_dataset reports rejections, the router should log a warning
    rather than swallow them silently."""
    from chap_core.rest_api.data_models import ValidationError
    from chap_core.rest_api.v1.routers import crud

    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Rejection log")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    def fake_validate(_feature_names, dataset):
        return dataset, [
            ValidationError(reason="missing data", org_unit="loc_x", feature_name="rainfall", time_periods=[])
        ]

    monkeypatch.setattr(crud, "validate_full_dataset", fake_validate)

    class _FakeJob:
        id = "captured-job"

    class _CapturingWorker:
        def queue_db(self, *args, **kwargs):
            return _FakeJob()

    monkeypatch.setattr(crud, "worker", _CapturingWorker())

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload["nPeriods"] = 3

    caplog.set_level(logging.WARNING, logger="chap_core.rest_api.v1.routers.crud")
    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 200, response.json()
    matching = [r for r in caplog.records if "observations rejected for prediction-setup" in r.getMessage()]
    assert matching, "expected rejection-count warning to be logged"


def test_delete_prediction_setup_redis_unavailable_returns_503(override_session, seeded_session, monkeypatch):
    """If Redis is unreachable we cannot sweep in-flight jobs that point at this setup,
    so the DELETE must fail loudly rather than risk an FK-violation on later prediction inserts."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Redis-down target")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    class _BrokenRedis:
        def keys(self, _pattern):
            raise RuntimeError("redis is down")

        def hgetall(self, _key):
            raise RuntimeError("redis is down")

        def delete(self, _key):
            raise RuntimeError("redis is down")

    from chap_core.rest_api.v1.routers import crud

    monkeypatch.setattr(crud, "redis", _BrokenRedis())

    response = client.delete(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 503, response.json()

    response = client.get(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 200, response.json()


def test_run_prediction_setup_normalizes_dataset_type_to_prediction(
    override_session, seeded_session, example_polygons, monkeypatch
):
    """Regression: chap-scheduler sends `type='forecasting'` on the wire, but the dataset
    must be persisted with type='prediction' so it shows up alongside other prediction-
    driven datasets in UI filters that key on the type column. Verified by intercepting
    the worker.queue_db call and inspecting the dataset_create_info it received."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = _create_prediction_setup(backtest.id, "Type normalize")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    captured: dict = {}

    class _FakeJob:
        id = "captured-job"

    class _CapturingWorker:
        def queue_db(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeJob()

    from chap_core.rest_api.v1.routers import crud

    monkeypatch.setattr(crud, "worker", _CapturingWorker())

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload["nPeriods"] = 3
    payload["type"] = "forecasting"  # what chap-scheduler currently sends; should be overridden server-side

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 200, response.json()

    dataset_info = captured["kwargs"]["dataset_create_info"]
    assert dataset_info["type"] == "prediction"


@pytest.mark.skip(
    reason=(
        "Pre-existing test-fixture mismatch: the seeded `backtest` fixture has model_id='naive_model' "
        "but model_db_id=1, which points at chap_auto_ewars (the first yaml-seeded R model), not the "
        "NaiveEstimator path. Running the actual celery job then attempts to invoke Rscript and fails. "
        "The end-to-end behavior is exercised by the docker-compose integration suite (make test-all), "
        "which uses a known-good dataset payload. Unskip once the fixture model_db_id is wired to the "
        "naive_model row (or once we have a Python-only model with covariate adapters that match the "
        "test data)."
    )
)
def test_run_prediction_setup_full_flow(
    celery_session_worker, clean_engine, dependency_overrides, example_polygons, dataset, backtest
):
    """End-to-end: create a setup, fire /run, await the celery job, and verify the
    resulting Prediction row has prediction_setup_id pointing back at the setup."""
    with Session(clean_engine) as session:
        session.add(dataset)
        session.add(backtest)
        session.commit()
        session.refresh(backtest)
        backtest_id = backtest.id
    assert backtest_id is not None

    model_list = client.get("/v1/crud/configured-models").json()
    models = [ModelSpecRead.model_validate(m) for m in model_list]
    model = next(m for m in models if m.name == "naive_model")
    # naive_model declares rainfall + mean_temperature, but the underlying ExternalModel
    # adapter also references population. Include it explicitly so _adapt_data finds it.
    provided_features = [f.name for f in model.covariates] + ["disease_cases", "population"]

    created = _create_prediction_setup(backtest_id, "End-to-end run")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    request = create_make_data_request(example_polygons, [], provided_features)
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload["nPeriods"] = 3

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 200, response.json()
    prediction_db_id = await_result_id(response.json()["id"])

    response = client.get(f"/v1/crud/predictions/{prediction_db_id}")
    assert response.status_code == 200, response.json()
    info = PredictionInfo.model_validate(response.json())
    assert info.prediction_setup_id == setup_id

    response = client.get(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 200, response.json()
    body = response.json()
    assert any(p["id"] == prediction_db_id for p in body["predictions"])


def _make_dataset(
    make_dataset_request,
    wanted_field_names=["rainfall", "disease_cases", "population", "mean_temperature"],
    expected_rejections=None,
):
    data = make_dataset_request.model_dump_json()
    response = client.post("/v1/analytics/make-dataset", content=data)
    content = response.json()
    _check_rejected_org_units(content, expected_rejections)

    assert response.status_code == 200, content
    db_id = await_result_id(content["id"])
    dataset_list = client.get("/v1/crud/datasets").json()
    assert len(dataset_list) > 0
    assert db_id in {ds["id"] for ds in dataset_list}
    response = client.get(f"/v1/crud/datasets/{db_id}")
    assert response.status_code == 200, response.json()
    ds = DataSetWithObservations.model_validate(response.json())
    population_entries = [o for o in ds.observations if o.feature_name == "population"]
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


def _check_rejected_org_units(content, expected_rejections):
    if expected_rejections is not None:
        assert "rejected" in content, content
        rejected_regions = {rejection["orgUnit"] for rejection in content["rejected"]}
        assert rejected_regions == set(expected_rejections), (rejected_regions, expected_rejections)


@pytest.mark.skip(reason="Failing because of missing geojson file")
def test_add_csv_dataset(celery_session_worker, dependency_overrides, data_path):
    csv_data = open(data_path / "nicaragua_weekly_data.csv", "rb")
    geojson_data = open(data_path / "nicaragua.json", "rb")
    response = client.post("/v1/crud/datasets/csvFile", files={"csvFile": csv_data, "geojsonFile": geojson_data})
    assert response.status_code == 200, response.json()


def test_full_prediction_flow(celery_session_worker, dependency_overrides, example_polygons):
    model_list = client.get("/v1/crud/configured-models").json()
    models = [ModelSpecRead.model_validate(m) for m in model_list]
    model = next(m for m in models if m.name == "naive_model")
    features = [f.name for f in model.covariates]
    fetched_feature, *provided_features = features
    provided_features = features + ["disease_cases"]
    data_sources = client.get("/v1/analytics/data-sources").json()
    data_source = next(ds for ds in data_sources if fetched_feature in ds["supportedFeatures"])
    fetch_request = []  # [FetchRequest(feature_name=fetched_feature, data_source_name=data_source["name"])]
    request = create_make_data_request(example_polygons, fetch_request, provided_features)
    request = MakePredictionRequest(model_id=model.name, **request.model_dump())
    data = request.model_dump(mode="json")
    response = client.post("/v1/analytics/make-prediction", json=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()["id"])
    response = client.get("/v1/crud/predictions")
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0
    p_infos = [PredictionInfo.model_validate(entry) for entry in response.json()]
    for p_info in p_infos:
        assert p_info.configured_model is not None
        assert p_info.configured_model.name
        assert p_info.dataset.data_sources is not None
    print(p_infos)

    response = client.get(f"/v1/analytics/prediction-entry/{db_id}", params={"quantiles": [0.1, 0.5, 0.9]})
    assert response.status_code == 200, response.json()
    ds = [PredictionEntry.model_validate(entry) for entry in response.json()]
    assert len(ds) > 0
    assert all(pe.quantile in (0.1, 0.5, 0.9) for pe in ds)


@pytest.mark.parametrize("dry_run", [False, True])
def test_backtest_with_data_flow(
    celery_session_worker, dependency_overrides, example_polygons, create_backtest_with_data_request, dry_run
):
    request_payload = create_backtest_with_data_request.model_dump()
    _check_backtest_with_data(request_payload, expected_rejections=[], dry_run=dry_run)


def test_backtest_with_empty_provided_data(dependency_overrides, create_backtest_with_data_request):
    request_payload = create_backtest_with_data_request.model_dump()
    request_payload["provided_data"] = []
    response = client.post("/v1/analytics/create-backtest-with-data", json=request_payload)
    assert response.status_code == 400
    assert "No observation data provided" in response.json()["detail"]


def test_backtest_with_empty_provided_data_dry_run(dependency_overrides, create_backtest_with_data_request):
    request_payload = create_backtest_with_data_request.model_dump()
    request_payload["provided_data"] = []
    response = client.post("/v1/analytics/create-backtest-with-data?dryRun=true", json=request_payload)
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["id"] is None
    assert body["importedCount"] == 0
    assert body["rejected"] == []


def test_backtest_with_all_regions_rejected_dry_run(dependency_overrides, create_backtest_with_data_request):
    request_payload = create_backtest_with_data_request.model_dump()
    obs = request_payload["provided_data"]
    target_feature = next(e["feature_name"] for e in obs if e["feature_name"] != "disease_cases")
    seen_locations = set()
    pruned = []
    for entry in obs:
        if entry["feature_name"] == target_feature and entry["org_unit"] not in seen_locations:
            seen_locations.add(entry["org_unit"])
            continue
        pruned.append(entry)
    request_payload["provided_data"] = pruned

    response = client.post("/v1/analytics/create-backtest-with-data?dryRun=true", json=request_payload)
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["id"] is None
    assert body["importedCount"] == 0
    assert len(body["rejected"]) > 0
    assert all(r["featureName"] == target_feature for r in body["rejected"])


@pytest.mark.parametrize("dry_run", [False, True])
def test_backtest_with_weekly_data_flow(
    celery_session_worker, dependency_overrides, example_polygons, create_backtest_with_weekly_data_request, dry_run
):
    request_payload = create_backtest_with_weekly_data_request.model_dump()
    print(request_payload)
    _check_backtest_with_data(request_payload, expected_rejections=[], dry_run=dry_run, expected_period_type="week")


@pytest.fixture()
def local_backtest_request(local_data_path):
    return json.load(open(local_data_path / "create-backtest-from-data.json", "r"))


# @pytest.mark.skip(reason="This ends up with an empty dataset")
def test_local_backtest_with_data(
    local_backtest_request, celery_session_worker, dependency_overrides, example_polygons
):
    url = "/v1/analytics/create-backtest-with-data"
    response = client.post(url, json=local_backtest_request)
    content = response.json()
    assert response.status_code == 500, content
    detail = content["detail"]
    assert "missing" in detail["message"].lower(), detail["message"].lower()
    assert len(detail["rejected"]) == 1


def _check_backtest_with_data(request_payload, expected_rejections=None, dry_run=False, expected_period_type="month"):
    url = "/v1/analytics/create-backtest-with-data"
    if dry_run:
        url += "?dryRun=true"
    response = client.post(url, json=request_payload)
    content = response.json()
    assert response.status_code == 200, content
    _check_rejected_org_units(content, expected_rejections)
    job_id = content["id"]
    if dry_run:
        assert job_id is None, "Job ID should be None for dry run"
        return
    db_id = await_result_id(job_id, timeout=180)
    response = client.get(f"/v1/crud/backtests/{db_id}/info")
    assert response.status_code == 200, response.json()
    backtest_info = BacktestRead.model_validate(response.json())
    assert len(backtest_info.dataset.data_sources) > 0, backtest_info.dataset
    assert len(backtest_info.dataset.org_units) > 0, backtest_info.dataset
    assert backtest_info.dataset.last_period is not None, backtest_info.dataset
    assert backtest_info.dataset.period_type == expected_period_type, backtest_info.dataset
    # assert len(backtest_info.metrics) > 0
    created_dataset_id = backtest_info.dataset_id
    dataset_response = client.get(f"/v1/crud/datasets/{created_dataset_id}")
    assert dataset_response.status_code == 200, dataset_response.json()
    eval_params = {"backtestId": db_id, "quantiles": [0.5]}
    eval_response = client.get("/v1/analytics/evaluation-entry", params=eval_params)
    assert eval_response.status_code == 200, eval_response.json()
    evaluation_entries = eval_response.json()
    assert len(evaluation_entries) > 0
    EvaluationEntry.model_validate(evaluation_entries[0])
    for plot_name in ["metric_by_horizon_mean", "metric_map"]:
        response = client.get(f"/v1/visualization/metric-plots/{plot_name}/{db_id}/crps")
        assert response.status_code == 200, response.json()


def test_add_configured_model_flow(celery_session_worker, dependency_overrides):
    url = "/v1/crud/model-templates"
    content = get_content(url)
    assert isinstance(content, list)
    for m in content:
        logger.info(m)

    model = next(m for m in content if m["name"] == "ewars_template")
    template_id = model["id"]
    print(template_id)
    config = ModelConfigurationCreate(
        name="testing",
        model_template_id=template_id,
        additional_continuous_covariates=["rainfall"],
        user_option_values=dict(precision=2.0, n_lags=3),
    )

    response = client.post("/v1/crud/configured-models", json=config.model_dump())
    assert response.status_code == 200, response.json()


def test_add_configured_model_unknown_template_returns_404(dependency_overrides):
    payload = {"name": "orphan", "modelTemplateId": 999999, "userOptionValues": {}}
    response = client.post("/v1/crud/configured-models", json=payload)
    assert response.status_code == 404, response.text


def test_add_configured_model_without_user_option_values(dependency_overrides):
    """Omitting userOptionValues should not 500; a template with no required user
    options should accept an empty configuration."""
    content = get_content("/v1/crud/model-templates")
    model = next(m for m in content if m["name"] == "naive_model")
    payload = {"name": "naive_without_options", "modelTemplateId": model["id"]}

    response = client.post("/v1/crud/configured-models", json=payload)
    assert response.status_code == 200, response.json()
    assert response.json()["userOptionValues"] == {}


def test_get_configured_model_info(celery_session_worker, dependency_overrides):
    configured = get_content("/v1/crud/configured-models")
    default = next(m for m in configured if m["name"] == "chap_ewars_monthly")

    response = client.get(f"/v1/crud/configured-models/{default['id']}")
    assert response.status_code == 200, response.json()
    body = response.json()
    for key in ("id", "name", "displayName", "modelTemplateId", "modelTemplate"):
        assert key in body, body.keys()
    info = ConfiguredModelInfoRead.model_validate(body)
    assert info.name == "chap_ewars_monthly"

    missing = client.get("/v1/crud/configured-models/999999")
    assert missing.status_code == 404, missing.json()


def get_content(url):
    response = client.get(url)
    content = response.json()
    assert response.status_code == 200, content
    return content


def test_all_backtest_plots_via_api(override_session, seeded_session):
    """Test that all registered backtest plots can be generated via the REST API."""
    # First, list all available backtest plot types
    response = client.get("/v1/visualization/backtest-plots/")
    assert response.status_code == 200, response.json()
    plot_types = response.json()
    assert len(plot_types) > 0, "No backtest plot types registered"

    # Get a backtest ID from the seeded database
    from chap_core.database.tables import Backtest
    from sqlmodel import select

    backtests = seeded_session.exec(select(Backtest)).all()
    assert len(backtests) > 0, "No backtests in seeded database"
    backtest_id = backtests[0].id

    # Test each plot type
    for plot_type in plot_types:
        plot_id = plot_type["id"]
        url = f"/v1/visualization/backtest-plots/{plot_id}/{backtest_id}"
        response = client.get(url)
        assert response.status_code == 200, f"Plot {plot_id} failed: {response.json()}"
        # Verify the response is valid Vega JSON
        vega_spec = response.json()
        assert "$schema" in vega_spec or "data" in vega_spec, f"Plot {plot_id} returned invalid Vega spec"


def test_get_dataset_csv(override_session, seeded_session):
    """Test that dataset can be downloaded as CSV."""
    datasets = seeded_session.exec(select(DataSet)).all()
    assert len(datasets) > 0, "No datasets in seeded database"
    dataset_id = datasets[0].id

    response = client.get(f"/v1/crud/datasets/{dataset_id}/csv")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "Content-Disposition" in response.headers
    assert f"filename=dataset_{dataset_id}.csv" in response.headers["Content-Disposition"]

    csv_content = response.text
    lines = csv_content.strip().split("\n")
    assert len(lines) > 1, "CSV should have header and data rows"
    header = lines[0]
    assert "time_period" in header, f"Expected time_period in header, got: {header}"


def test_get_dataset_df_unknown_id_returns_404(clean_engine, dependency_overrides):
    response = client.get("/v1/crud/datasets/999999/df")
    assert response.status_code == 404, response.text


def test_get_dataset_csv_unknown_id_returns_404(clean_engine, dependency_overrides):
    response = client.get("/v1/crud/datasets/999999/csv")
    assert response.status_code == 404, response.text


def test_get_dataset_df_with_nans(override_session, seeded_session):
    """Datasets containing NaN observations must round-trip through /df as JSON.
    Previously pandas NaN floats leaked into the response and triggered a 500
    because they are not JSON-serialisable."""
    dataset = seeded_session.exec(select(DataSet).where(DataSet.name == "dataset_with_nans")).one()

    response = client.get(f"/v1/crud/datasets/{dataset.id}/df")
    assert response.status_code == 200, response.text

    records = response.json()
    assert len(records) > 0
    assert any(record.get("disease_cases") is None for record in records), (
        "Expected at least one None disease_cases value in the response"
    )
