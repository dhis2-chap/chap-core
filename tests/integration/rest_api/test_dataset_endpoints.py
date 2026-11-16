import json

import numpy as np
import pandas as pd
import pytest
from sqlmodel import select

from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import (
    DataSet,
    DataSetWithObservations,
    FeatureCollectionModel,
    MaskedRange,
)
from chap_core.database.tables import (
    Backtest,
    Prediction,
)
from chap_core.datatypes import HealthPopulationData, create_tsdataclass
from chap_core.rest_api.data_models import (
    BacktestCreate,
    DatasetCreate,
)
from chap_core.rest_api.db_worker_functions import (
    harmonize_and_add_dataset,
    harmonize_and_add_health_dataset,
    run_backtest,
    run_prediction,
)
from chap_core.spatio_temporal_data.converters import observations_to_dataset
from tests.integration.rest_api.db_endpoint_helpers import (
    await_result_id,
    check_all_regions_rejected_dry_run,
    check_rejected_org_units,
    client,
)


def test_add_dataset_flow(celery_session_worker, dependency_overrides, dataset_create: DatasetCreate):
    data = dataset_create.model_dump(mode="json")
    data["masked_ranges"] = [{"start": "2021-06", "end": "2021-08", "name": "outage"}]
    print(json.dumps(data, indent=2))
    response = client.post("/v1/crud/datasets", json=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()["id"])
    response = client.get(f"/v1/crud/datasets/{db_id}")
    assert response.status_code == 200, response.json()
    ds = DataSetWithObservations.model_validate(response.json())
    assert response.json()["maskedRanges"] == [{"start": "202106", "end": "202108", "name": "outage"}]

    assert len(ds.observations) > 0
    assert ds.created_manually
    print(response.json())
    assert "orgUnit" in response.json()["observations"][0], response.json()["observations"][0].keys()


def test_make_dataset_import_persists_data_sources(clean_engine, dataset_make_request):
    """The covariate to data-element mapping sent to make-dataset must survive the import."""
    request = dataset_make_request
    assert request.data_sources, "fixture should provide a mapping to persist"
    feature_names = list({obs.feature_name for obs in request.provided_data})
    provided_data = observations_to_dataset(create_tsdataclass(feature_names), request.provided_data, fill_missing=True)
    provided_data.set_polygons(request.geojson)

    with SessionWrapper(clean_engine) as session:
        dataset_id = harmonize_and_add_dataset(
            feature_names,
            request.data_to_be_fetched,
            provided_data.model_dump(),
            request.name,
            request.type,
            session=session,
            data_sources=request.data_sources,
        )
        stored = session.session.get(DataSet, dataset_id)
        assert stored is not None
        assert stored.data_sources == request.data_sources
        assert stored.created_manually


def test_make_dataset_import_persists_masked_ranges(clean_engine, dataset_make_request):
    """Masked ranges sent to make-dataset must survive the import (CLIM-1300)."""
    request = dataset_make_request
    feature_names = list({obs.feature_name for obs in request.provided_data})
    provided_data = observations_to_dataset(create_tsdataclass(feature_names), request.provided_data, fill_missing=True)
    provided_data.set_polygons(request.geojson)
    masked_ranges = [MaskedRange(start="2021-06", end="2021-08")]

    with SessionWrapper(clean_engine) as session:
        dataset_id = harmonize_and_add_dataset(
            feature_names,
            request.data_to_be_fetched,
            provided_data.model_dump(),
            request.name,
            request.type,
            session=session,
            masked_ranges=masked_ranges,
        )
        stored = session.session.get(DataSet, dataset_id)
        assert stored is not None
        assert stored.masked_ranges == masked_ranges


def test_health_dataset_import_persists_masked_ranges(clean_engine, dataset_create):
    """Masked ranges sent to POST /v1/crud/datasets must survive the import (CLIM-1300)."""
    health_data = observations_to_dataset(HealthPopulationData, dataset_create.observations, fill_missing=True)
    health_data.set_polygons(FeatureCollectionModel.model_validate(dataset_create.geojson.model_dump()))
    masked_ranges = [MaskedRange(start="2021-06", end="2021-08")]

    with SessionWrapper(clean_engine) as session:
        dataset_id = harmonize_and_add_health_dataset(
            health_data.model_dump(), dataset_create.name, session=session, masked_ranges=masked_ranges
        )
        stored = session.session.get(DataSet, dataset_id)
        assert stored is not None
        assert stored.masked_ranges == masked_ranges


def test_get_data_sources():
    response = client.get("/v1/analytics/data-sources")
    data = response.json()
    assert response.status_code == 200, data
    assert len(data) == 9, data
    assert next(ds for ds in data if "rainfall" in ds["supportedFeatures"])["dataset"] == "era5"


def test_get_covariate_names(dependency_overrides):
    response = client.get("/v1/analytics/covariate-names")
    data = response.json()
    assert response.status_code == 200, data
    by_name = {entry["name"]: entry for entry in data}
    assert len(by_name) == len(data), "names should be unique"
    assert by_name["rainfall"]["standard"] is True
    assert "naive_model" in by_name["rainfall"]["requiredBy"]
    assert by_name["disease_cases"]["standard"] is True
    assert "naive_model" in by_name["disease_cases"]["requiredBy"], "target column should be suggested too"
    assert "time_period" not in by_name
    assert "location" not in by_name
    assert all(entry["standard"] or entry["requiredBy"] for entry in data)

    configured_models = client.get("/v1/crud/configured-models").json()
    extras = {
        (model["name"], covariate)
        for model in configured_models
        for covariate in model["additionalContinuousCovariates"]
    }
    assert extras, "seeded config should include a model with extra covariates"
    for model_name, covariate in extras:
        assert covariate in by_name, (covariate, sorted(by_name))
        assert model_name in by_name[covariate]["requiredBy"], by_name[covariate]


def test_dataset_target_is_saved_and_used_by_runs(
    override_session, p_seeded_engine, dataset_make_request, baseline_worker, monkeypatch
):
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.rest_api import db_worker_functions
    from chap_core.rest_api.v1.routers import analytics

    trained_on = []
    forecast_ahead = db_worker_functions.forecast_ahead

    def record_forecast(estimator, dataset, *args, **kwargs):
        trained_on.append(dataset)
        return forecast_ahead(estimator, dataset, *args, **kwargs)

    monkeypatch.setattr(db_worker_functions, "forecast_ahead", record_forecast)
    monkeypatch.setattr(analytics, "worker", baseline_worker)
    request = dataset_make_request
    request.target_column = "rainfall"
    for observation in request.provided_data:
        if observation.feature_name == "rainfall":
            observation.value = None if observation.period == "2020-01" else 7.0
        # Gaps in the target and in disease_cases, which runs replace with the target, reject nothing.
        if observation.feature_name == "disease_cases" and observation.period == "2020-02":
            observation.value = None
    response = client.post("/v1/analytics/make-dataset", json=request.model_dump(mode="json", by_alias=True))
    assert response.status_code == 200, response.text
    assert response.json()["rejected"] == []
    func, args, kwargs = baseline_worker.queued.pop()
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = func(
            *args, session=session, target_column=kwargs["target_column"], data_sources=kwargs["data_sources"]
        )
        # Worker callers which omit the target also inherit it from the dataset.
        backtest_id = run_backtest(
            BacktestCreate(name="rain", dataset_id=dataset_id, model_id="naive_model"),
            n_periods=3,
            n_splits=2,
            session=session,
        )
        backtest = session.session.get(Backtest, backtest_id)
        assert backtest.specification.target_column == "rainfall"
        assert set(Evaluation.from_backtest(backtest).to_flat().observations.disease_cases) == {7.0}
        prediction_id = run_prediction("naive_model", str(dataset_id), 3, "rain", session)
        prediction = session.session.get(Prediction, prediction_id)
        assert prediction.forecasts
        assert "rainfall" not in trained_on[0].field_names()
        for data in trained_on[0].values():
            assert set(data.disease_cases[~np.isnan(data.disease_cases)]) == {7.0}

    assert client.get(f"/v1/crud/datasets/{dataset_id}").json()["targetColumn"] == "rainfall"
    rows = client.get("/v1/crud/datasets").json()
    assert next(row for row in rows if row["id"] == dataset_id)["targetColumn"] == "rainfall"
    actual = client.get(f"/v1/analytics/actual-cases/{dataset_id}", params={"isDatasetId": True}).json()
    assert actual["featureId"] == actual["dhis2Id"] == "rainfall"
    assert {entry["value"] for entry in actual["data"] if entry["value"] is not None} == {7.0}
    thresholds = client.post(
        "/v1/analytics/thresholds",
        json={"datasetId": dataset_id, "periodIds": ["2023-01"], "params": {"type": "seasonal"}},
    )
    assert thresholds.status_code == 200, thresholds.text
    assert all(entry["values"] == [7.0] for entry in thresholds.json()["entries"])

    payload = {"name": "rain", "datasetId": dataset_id, "modelId": "naive_model", "modelIds": ["naive_model"]}
    for endpoint in ("create-backtest", "create-backtests"):
        for target in ({}, {"targetColumn": "rainfall"}):
            response = client.post(f"/v1/analytics/{endpoint}", json=payload | target)
            assert response.status_code == 200, response.text
            assert baseline_worker.queued[-1][2]["target_column"] == "rainfall"
        baseline_worker.queued.clear()
        response = client.post(f"/v1/analytics/{endpoint}", json=payload | {"targetColumn": "disease_cases"})
        assert response.status_code == 422, response.text
        assert "create a new dataset" in response.text
        assert baseline_worker.queued == []

    # A saved target the dataset has no column for, as a migrated dataset can have, is rejected too.
    with SessionWrapper(p_seeded_engine) as session:
        stored = session.session.get(DataSet, dataset_id)
        stored.target_column = "missing"
        session.session.add(stored)
        session.session.commit()
    response = client.post("/v1/analytics/create-backtest", json=payload)
    assert response.status_code == 422, response.text
    assert "has no target column 'missing'" in response.text


@pytest.mark.parametrize(
    "target_column,masked_ranges", [("missing", []), ("rainfall", [{"start": "202106", "end": "202108"}])]
)
def test_make_dataset_rejects_invalid_target_before_queueing(
    dependency_overrides, dataset_make_request, baseline_worker, monkeypatch, target_column, masked_ranges
):
    from chap_core.rest_api.v1.routers import analytics

    monkeypatch.setattr(analytics, "worker", baseline_worker)
    payload = dataset_make_request.model_dump(mode="json", by_alias=True)
    payload.update(targetColumn=target_column, maskedRanges=masked_ranges)
    response = client.post("/v1/analytics/make-dataset", json=payload)
    assert response.status_code == 422, response.text
    assert baseline_worker.queued == []


def test_make_dataset_rejects_masked_ranges_without_disease_cases(dependency_overrides, dataset_make_request):
    payload = dataset_make_request.model_dump(mode="json")
    payload["provided_data"] = [o for o in payload["provided_data"] if o["feature_name"] != "disease_cases"]
    payload["masked_ranges"] = [{"start": "2021-06", "end": "2021-08"}]
    response = client.post("/v1/analytics/make-dataset?dryRun=true", json=payload)
    assert response.status_code == 422, response.text
    assert "has no target column 'disease_cases'" in response.text


def test_make_dataset_rejects_input_carrying_the_mask_indicator(dependency_overrides, dataset_make_request):
    payload = dataset_make_request.model_dump(mode="json")
    first = payload["provided_data"][0]
    payload["provided_data"].append({**first, "feature_name": "target_masked", "value": 1.0})
    response = client.post("/v1/analytics/make-dataset?dryRun=true", json=payload)
    assert response.status_code == 422, response.text
    assert "is reserved" in response.text


def _as_columns(payload: dict) -> dict:
    """The same make-dataset payload with its observations sent as `providedDataColumns`."""
    wide = pd.DataFrame(payload.pop("providedData")).pivot(
        index=["orgUnit", "period"], columns="featureName", values="value"
    )
    wide = wide.astype(object).where(wide.notna(), None).reset_index()
    payload["providedDataColumns"] = {
        "orgUnit": wide["orgUnit"].tolist(),
        "period": wide["period"].tolist(),
        "values": {name: wide[name].tolist() for name in wide.columns if name not in ("orgUnit", "period")},
    }
    return payload


def test_make_dataset_columns_import_the_same_as_observations(
    dependency_overrides, dataset_make_request, baseline_worker, monkeypatch
):
    from chap_core.rest_api.v1.routers import analytics

    monkeypatch.setattr(analytics, "worker", baseline_worker)
    payload = dataset_make_request.model_dump(mode="json", by_alias=True)
    rainfall = next(o for o in payload["providedData"] if o["featureName"] == "rainfall")
    rainfall["value"] = None  # one rejected location, so the rejection summary is compared too
    responses = []
    for body in (payload, _as_columns(json.loads(json.dumps(payload)))):
        response = client.post("/v1/analytics/make-dataset", json=body)
        assert response.status_code == 200, response.text
        responses.append({key: value for key, value in response.json().items() if key != "id"})
    assert responses[0] == responses[1]
    assert responses[0]["rejected"]
    (_, from_columns, _), (_, from_observations, _) = baseline_worker.queued.pop(), baseline_worker.queued.pop()
    assert from_columns[2] == from_observations[2]


def test_make_dataset_rejects_both_observation_formats(dependency_overrides, dataset_make_request):
    payload = dataset_make_request.model_dump(mode="json", by_alias=True)
    payload["providedDataColumns"] = _as_columns(dataset_make_request.model_dump(mode="json", by_alias=True))[
        "providedDataColumns"
    ]
    response = client.post("/v1/analytics/make-dataset?dryRun=true", json=payload)
    assert response.status_code == 422, response.text
    assert "not both" in response.text


def test_make_dataset_rejects_uneven_columns(dependency_overrides, dataset_make_request):
    payload = _as_columns(dataset_make_request.model_dump(mode="json", by_alias=True))
    payload["providedDataColumns"]["period"].pop()
    response = client.post("/v1/analytics/make-dataset?dryRun=true", json=payload)
    assert response.status_code == 422, response.text
    assert "Every column must have" in response.text


def test_make_dataset_rejects_duplicate_rows_in_columns(dependency_overrides, dataset_make_request):
    payload = _as_columns(dataset_make_request.model_dump(mode="json", by_alias=True))
    columns = payload["providedDataColumns"]
    for column in [columns["orgUnit"], columns["period"], *columns["values"].values()]:
        column.append(column[0])
    response = client.post("/v1/analytics/make-dataset?dryRun=true", json=payload)
    assert response.status_code == 422, response.text
    assert "more than one row" in response.text


@pytest.mark.parametrize(
    "weekly, other_spelling",
    [
        (True, lambda period: period.replace("W", "W0")),  # 2020W1 and 2020W01
        (True, lambda period: period.replace("W", "-W")),  # 2020W1 and 2020-W1
        (False, lambda period: period.replace("-", "")),  # 2020-01 and 202001
    ],
)
def test_make_dataset_rejects_two_spellings_of_one_period_in_columns(
    dependency_overrides, dataset_make_request, weekly, other_spelling
):
    payload = _as_columns(dataset_make_request.model_dump(mode="json", by_alias=True))
    columns = payload["providedDataColumns"]
    if weekly:
        columns["period"] = [f"{period[:4]}W{int(period[5:])}" for period in columns["period"]]  # 2020-01 -> 2020W1
    for column in [columns["orgUnit"], *columns["values"].values()]:
        column.append(column[0])
    columns["period"].append(other_spelling(columns["period"][0]))
    response = client.post("/v1/analytics/make-dataset?dryRun=true", json=payload)
    assert response.status_code == 422, response.text
    assert "more than one row" in response.text


@pytest.mark.parametrize("name", ["location", "time_period"])
def test_make_dataset_rejects_a_column_feature_named_like_the_row_keys(
    dependency_overrides, dataset_make_request, name
):
    payload = _as_columns(dataset_make_request.model_dump(mode="json", by_alias=True))
    values = payload["providedDataColumns"]["values"]
    values[name] = values["rainfall"]
    response = client.post("/v1/analytics/make-dataset?dryRun=true", json=payload)
    assert response.status_code == 422, response.text
    assert f"feature named {name}" in response.text


@pytest.mark.parametrize(
    "endpoint, extra",
    [
        ("create-backtest-with-data/", {"n_periods": 3, "n_splits": 2, "stride": 1}),
        ("make-prediction", {"n_periods": 3}),
    ],
)
def test_inline_data_routes_reject_masked_ranges(dependency_overrides, dataset_make_request, endpoint, extra):
    """Masked ranges belong to a dataset: they are set with make-dataset or POST /v1/crud/datasets only."""
    payload = dataset_make_request.model_dump(mode="json") | {"model_id": "naive_model", **extra}
    payload["masked_ranges"] = [{"start": "2021-06", "end": "2021-08"}]
    response = client.post(f"/v1/analytics/{endpoint}?dryRun=true", json=payload)
    assert response.status_code == 422, response.text
    assert "maskedRanges can only be set when a dataset is created" in response.text


def _make_dataset(
    make_dataset_request,
    wanted_field_names=["rainfall", "disease_cases", "population", "mean_temperature"],
    expected_rejections=None,
):
    data = make_dataset_request.model_dump_json()
    response = client.post("/v1/analytics/make-dataset", content=data)
    content = response.json()
    check_rejected_org_units(content, expected_rejections)

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


def test_add_csv_dataset(dependency_overrides, data_path):
    """The numeric columns are stored with the chosen target; a text column such as `parent` is ignored."""
    frame = pd.read_csv(data_path / "vietnam_monthly.csv")
    frame["parent"] = "vietnam"
    with open(data_path / "vietnam_monthly.geojson", "rb") as geojson_data:
        response = client.post(
            "/v1/crud/datasets/csvFile",
            files={"csv_file": ("with_parent.csv", frame.to_csv(index=False).encode()), "geojson_file": geojson_data},
            data={"targetColumn": "rainfall"},
        )
    assert response.status_code == 200, response.json()
    response = client.get(f"/v1/crud/datasets/{response.json()['id']}")
    assert response.status_code == 200, response.json()
    stored = DataSetWithObservations.model_validate(response.json())
    assert stored.created_manually
    assert stored.target_column == "rainfall"
    assert set(stored.covariates) == {"rainfall", "mean_temperature", "disease_cases", "population"}


def test_add_csv_dataset_stores_it_unmasked(dependency_overrides, data_path):
    """The CSV upload takes no masked ranges: every disease_cases value is stored as given."""
    with (
        open(data_path / "vietnam_monthly.csv", "rb") as csv_data,
        open(data_path / "vietnam_monthly.geojson", "rb") as geojson_data,
    ):
        response = client.post("/v1/crud/datasets/csvFile", files={"csv_file": csv_data, "geojson_file": geojson_data})
    assert response.status_code == 200, response.json()
    dataset = DataSetWithObservations.model_validate(client.get(f"/v1/crud/datasets/{response.json()['id']}").json())
    assert dataset.masked_ranges == []
    assert "target_masked" not in dataset.covariates
    stored_cases = [o.value for o in dataset.observations if o.feature_name == "disease_cases"]
    source_cases = pd.read_csv(data_path / "vietnam_monthly.csv")["disease_cases"]
    assert len(stored_cases) == len(source_cases)
    assert sum(v is None or np.isnan(v) for v in stored_cases) == source_cases.isna().sum()


def test_create_dataset_rejects_a_target_outside_the_health_fields(dependency_overrides, dataset_create):
    payload = dataset_create.model_dump(mode="json") | {"target_column": "rainfall"}
    response = client.post("/v1/crud/datasets", json=payload)
    assert response.status_code == 422, response.text
    assert "has no target column 'rainfall'" in response.text


def test_create_dataset_rejects_input_carrying_the_mask_indicator(dependency_overrides, dataset_create):
    payload = dataset_create.model_dump(mode="json")
    first = payload["observations"][0]
    payload["observations"].append({**first, "feature_name": "target_masked", "value": 1.0})
    response = client.post("/v1/crud/datasets", json=payload)
    assert response.status_code == 422, response.text
    assert "is reserved" in response.text


def test_add_csv_dataset_rejects_the_mask_indicator(dependency_overrides, data_path, tmp_path):
    df = pd.read_csv(data_path / "vietnam_monthly.csv")
    df["target_masked"] = 0
    csv_path = tmp_path / "with_indicator.csv"
    df.to_csv(csv_path, index=False)
    n_datasets = len(client.get("/v1/crud/datasets").json())
    with open(csv_path, "rb") as csv_data, open(data_path / "vietnam_monthly.geojson", "rb") as geojson_data:
        response = client.post("/v1/crud/datasets/csvFile", files={"csv_file": csv_data, "geojson_file": geojson_data})
    assert response.status_code == 422, response.text
    assert "is reserved" in response.text
    assert len(client.get("/v1/crud/datasets").json()) == n_datasets


def test_make_dataset_dry_run(dependency_overrides, dataset_make_request, org_units):
    response = client.post("/v1/analytics/make-dataset?dryRun=true", json=dataset_make_request.model_dump())
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["id"] is None
    assert body["importedCount"] == len(org_units)
    assert body["rejected"] == []


def test_make_dataset_all_regions_rejected_dry_run(dependency_overrides, dataset_make_request):
    check_all_regions_rejected_dry_run("/v1/analytics/make-dataset", dataset_make_request.model_dump())


def test_make_dataset_all_polygons_missing_returns_400_with_rejections(
    dependency_overrides, dataset_make_request, org_units
):
    request_payload = dataset_make_request.model_dump()
    request_payload["geojson"]["features"] = []
    response = client.post("/v1/analytics/make-dataset", json=request_payload)
    assert response.status_code == 400, response.json()
    detail = response.json()["detail"]
    assert detail["imported_count"] == 0
    assert {r["orgUnit"] for r in detail["rejected"]} == set(org_units)
    assert all(r["featureName"] == "polygon" for r in detail["rejected"])


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
