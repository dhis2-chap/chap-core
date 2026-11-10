from datetime import datetime, timedelta

import numpy as np
import pytest
from sqlmodel import Session, select

import chap_core
from chap_core.api_types import DataList, EvaluationEntry
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import (
    DataSet,
    Observation,
)
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateRole
from chap_core.database.tables import (
    Backtest,
    BacktestRead,
    BacktestSpecification,
)
from chap_core.rest_api.data_models import (
    BacktestCreate,
)
from chap_core.rest_api.db_worker_functions import (
    run_backtest,
)
from chap_core.time_period import TimePeriod
from tests.integration.rest_api.db_endpoint_helpers import (
    await_result_id,
    check_all_regions_rejected_dry_run,
    check_rejected_org_units,
    client,
    get_content,
    mask_seeded_dataset,
)


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


def test_compatible_backtests(clean_engine, dependency_overrides):
    with Session(clean_engine) as session:
        dataset = DataSet(name="ds", type="testing", created=datetime.now(), covariates=[])
        session.add(dataset)
        session.commit()

        ds_id = dataset.id
        # One specification for all three: same dataset, same (default) parameters.
        specification = BacktestSpecification(dataset_id=ds_id)
        session.add(specification)
        session.commit()
        backtest = Backtest(
            specification=specification,
            dataset_id=ds_id,
            name="testing",
            model_id="naive_model",
            model_db_id=1,
            org_units=["Oslo", "Bergen"],
            split_periods=["202201", "202202"],
        )
        matching = Backtest(
            specification=specification,
            dataset_id=ds_id,
            name="testing2",
            model_id="chap_auto_ewars",
            model_db_id=1,
            org_units=["Bergen", "Trondheim"],
            split_periods=["202202", "202203"],
        )
        non_matching = Backtest(
            specification=specification,
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
        ("nRetrain", 0),
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
    """/v1/analytics/create-backtest should reject bogus dataset ids synchronously
    rather than queueing a job that fails later."""
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


def test_create_backtest_unknown_model_returns_404_without_queueing(override_session, seeded_session, monkeypatch):
    """An unknown modelId is rejected before a job is queued, and the error names it (CLIM-733)."""
    from chap_core.rest_api.v1.routers import analytics

    queued: list = []

    class _RecordingWorker:
        def queue_db(self, func, info, **kwargs):
            queued.append(info)

    monkeypatch.setattr(analytics, "worker", _RecordingWorker())
    dataset_id = seeded_session.exec(select(DataSet.id)).first()

    payload = {"name": "x", "datasetId": dataset_id, "modelId": "no-such-model"}
    response = client.post("/v1/analytics/create-backtest", json=payload)
    assert response.status_code == 404, response.text
    assert "no-such-model" in response.json()["detail"]
    assert queued == []


def test_create_backtest_forwards_params_and_accepts_int_model_id(override_session, seeded_session, monkeypatch):
    """The endpoint must hand every BacktestParams field, including n_retrain, to the worker
    and accept the configured model's integer id as well as its name."""
    from chap_core.rest_api.v1.routers import analytics

    captured: dict = {}

    class _FakeJob:
        id = "captured-job"

    class _CapturingWorker:
        def queue_db(self, func, info, **kwargs):
            captured["info"] = info
            captured.update(kwargs)
            return _FakeJob()

    monkeypatch.setattr(analytics, "worker", _CapturingWorker())
    model = seeded_session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.name == "naive_model")).first()
    dataset_id = seeded_session.exec(select(DataSet.id)).first()

    payload = {"name": "x", "datasetId": dataset_id, "modelId": model.id, "nSplits": 4, "nRetrain": 2}
    response = client.post("/v1/analytics/create-backtest", json=payload)
    assert response.status_code == 200, response.text
    assert captured["info"].model_id == model.id
    assert (captured["n_periods"], captured["n_splits"], captured["stride"], captured["n_retrain"]) == (3, 4, 1, 2)


def test_deleted_configured_model_cannot_start_a_backtest(override_session, seeded_session, monkeypatch):
    """A deleted configured model stays listed as archived, but new runs on it are rejected (CLIM-1200)."""
    from chap_core.rest_api.v1.routers import analytics

    queued: list = []

    class _RecordingWorker:
        def queue_db(self, func, info, **kwargs):
            queued.append(info)

    monkeypatch.setattr(analytics, "worker", _RecordingWorker())
    model = seeded_session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.name == "naive_model")).first()
    dataset_id = seeded_session.exec(select(DataSet.id)).first()

    response = client.delete(f"/v1/crud/configured-models/{model.id}")
    assert response.status_code == 200, response.text

    # The Modeling App hides archived models itself and still needs them to name old runs.
    listed = next(m for m in get_content("/v1/crud/configured-models") if m["id"] == model.id)
    assert listed["archived"] is True

    payload = {"name": "x", "datasetId": dataset_id, "modelId": model.id}
    response = client.post("/v1/analytics/create-backtest", json=payload)
    assert response.status_code == 409, response.text
    assert "archived" in response.json()["detail"]
    assert queued == []


class _JobIdWorker:
    """Records every queued job and hands out distinct job ids, so a test can check what was submitted."""

    def __init__(self):
        self.queued: list[tuple] = []

    def queue_db(self, func, info, **kwargs):
        self.queued.append((func, info, kwargs))

        class _Job:
            id = f"job-{len(self.queued)}"

        return _Job()


def test_create_backtests_resolves_the_specification_and_queues_one_job_per_model(
    override_session, seeded_session, p_seeded_engine, monkeypatch
):
    """The response must name the specification the jobs will file under before any job has run,
    and the benchmarking server must get one job id per model without a second lookup."""
    from chap_core.rest_api.v1.routers import analytics

    worker = _JobIdWorker()
    monkeypatch.setattr(analytics, "worker", worker)
    naive = seeded_session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.name == "naive_model")).one()
    other = seeded_session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.name != "naive_model")).first()
    dataset_id = seeded_session.exec(select(DataSet.id)).first()

    payload = {
        "name": "bench",
        "datasetId": dataset_id,
        "modelIds": ["naive_model", other.id],
        "nSplits": 4,
        "nRetrain": 2,
    }
    response = client.post("/v1/analytics/create-backtests", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert [job["configuredModelId"] for job in data["jobs"]] == [naive.id, other.id]
    assert [job["jobId"] for job in data["jobs"]] == ["job-1", "job-2"]

    assert [info.name for _, info, _ in worker.queued] == [f"bench/{naive.name}", f"bench/{other.name}"]
    assert [info.model_id for _, info, _ in worker.queued] == [naive.id, other.id]
    for func, _, kwargs in worker.queued:
        assert func is run_backtest
        assert (kwargs["n_periods"], kwargs["n_splits"], kwargs["stride"], kwargs["n_retrain"]) == (3, 4, 1, 2)
        assert kwargs["queue_baselines"] is True
        assert "database_url" not in kwargs

    specification = client.get(f"/v1/crud/backtest-specifications/{data['specificationId']}").json()
    assert (specification["dataset"]["id"], specification["nSplits"], specification["nRetrain"]) == (dataset_id, 4, 2)
    assert specification["backtests"] == []
    # A worker running one of the queued jobs files its backtest under the returned id.
    with SessionWrapper(p_seeded_engine) as session:
        backtest_id = _run(session, f"bench/{naive.name}", dataset_id, n_periods=3, n_splits=4, stride=1, n_retrain=2)
        assert session.session.get(Backtest, backtest_id).specification_id == data["specificationId"]


def test_create_backtests_rejects_unknown_dataset_or_model_before_queueing(
    override_session, seeded_session, monkeypatch
):
    from chap_core.rest_api.v1.routers import analytics

    worker = _JobIdWorker()
    monkeypatch.setattr(analytics, "worker", worker)
    dataset_id = seeded_session.exec(select(DataSet.id)).first()

    response = client.post(
        "/v1/analytics/create-backtests", json={"name": "x", "datasetId": 999999, "modelIds": ["naive_model"]}
    )
    assert response.status_code == 404, response.text
    response = client.post(
        "/v1/analytics/create-backtests",
        json={"name": "x", "datasetId": dataset_id, "modelIds": ["naive_model", "no_such_model"]},
    )
    assert response.status_code == 404, response.text
    assert "no_such_model" in response.json()["detail"]
    response = client.post(
        "/v1/analytics/create-backtests", json={"name": "x", "datasetId": dataset_id, "modelIds": []}
    )
    assert response.status_code == 422, response.text
    assert worker.queued == []


def test_create_backtests_rejects_a_window_that_leaves_no_org_unit_to_train_on(
    override_session, seeded_session, monkeypatch
):
    """The endpoint filters the dataset before queueing, so it can refuse up front instead of
    creating an empty specification and one doomed job per model."""
    from chap_core.rest_api.v1.routers import analytics

    worker = _JobIdWorker()
    monkeypatch.setattr(analytics, "worker", worker)
    dataset_id = seeded_session.exec(select(DataSet.id)).first()

    payload = {"name": "x", "datasetId": dataset_id, "modelIds": ["naive_model"], "nSplits": 1000}
    response = client.post("/v1/analytics/create-backtests", json=payload)
    assert response.status_code == 422, response.text
    assert "No org unit" in response.json()["detail"]
    assert worker.queued == []
    assert (
        client.get("/v1/crud/backtest-specifications", params={"datasetId": dataset_id, "nSplits": 1000}).json() == []
    )


def test_run_backtest_persists_resolved_params(override_session, p_seeded_engine):
    """The stored row records the parameters that actually ran, and an integer model id is resolved to its name."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        model = session.get_configured_model_by_id_or_name("naive_model")
        backtest_id = run_backtest(
            BacktestCreate(name="params", dataset_id=dataset_id, model_id=model.id),
            n_periods=3,
            n_splits=2,
            stride=1,
            n_retrain=2,
            session=session,
        )
        row = session.session.get(Backtest, backtest_id)
        assert (row.n_periods, row.n_splits, row.stride, row.n_retrain) == (3, 2, 1, 2)
        assert row.model_id == "naive_model"

    response = client.get(f"/v1/crud/backtests/{backtest_id}")
    assert response.status_code == 200, response.text
    read = BacktestRead.model_validate(response.json())
    assert (read.n_periods, read.n_splits, read.stride, read.n_retrain) == (3, 2, 1, 2)


def test_run_backtest_persists_the_future_weather_provider(override_session, p_seeded_engine):
    """A non-default provider must reach the specification, or a look-ahead run is filed as climatology."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        model = session.get_configured_model_by_id_or_name("naive_model")
        backtest_id = run_backtest(
            BacktestCreate(name="provider", dataset_id=dataset_id, model_id=model.id),
            n_periods=3,
            n_splits=2,
            stride=1,
            session=session,
            future_weather_provider="observed",
        )
        assert session.session.get(Backtest, backtest_id).future_weather_provider == "observed"

    response = client.get(f"/v1/crud/backtests/{backtest_id}")
    assert response.status_code == 200, response.text
    assert BacktestRead.model_validate(response.json()).future_weather_provider == "observed"


def _run(session, name, dataset_id, **params):
    return run_backtest(
        BacktestCreate(name=name, dataset_id=dataset_id, model_id="naive_model"), session=session, **params
    )


def test_backtest_created_carries_a_utc_offset(override_session, p_seeded_engine):
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest_id = _run(session, "utc", dataset_id, n_periods=3, n_splits=2, stride=1, n_retrain=1)

    response = client.get(f"/v1/crud/backtests/{backtest_id}")
    assert response.status_code == 200, response.text
    assert datetime.fromisoformat(response.json()["created"]).utcoffset() == timedelta(0)


def test_run_backtest_records_the_train_cutoff_not_the_dataset_end(p_seeded_engine):
    """`last_train_period` is the end of the training window, so it must precede every
    period forecast from it. Using the end of the full dataset instead puts the cutoff
    after the held-out window and makes horizon distances come out wrong."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest_id = _run(session, "cutoff", dataset_id, n_periods=3, n_splits=2, stride=1)
        forecasts = session.session.get(Backtest, backtest_id).forecasts
        assert forecasts
        assert all(TimePeriod.parse(f.last_train_period) < TimePeriod.parse(f.period) for f in forecasts), (
            "last_train_period must be before the forecast period"
        )


def test_backtests_with_the_same_parameters_share_one_specification(p_seeded_engine):
    """Deduplication is what makes "these two results are comparable" structural."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        params = {"n_periods": 3, "n_splits": 2, "stride": 1, "n_retrain": 1}
        first = session.session.get(Backtest, _run(session, "first", dataset_id, **params))
        second = session.session.get(Backtest, _run(session, "second", dataset_id, **params))

        assert first.specification_id == second.specification_id
        assert (first.n_periods, first.n_splits, first.stride, first.n_retrain) == (3, 2, 1, 1)


def test_backtest_records_the_chap_version_that_produced_it(p_seeded_engine):
    """Without this a metric shift over time cannot be attributed to the model or the platform."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest = session.session.get(Backtest, _run(session, "versioned", dataset_id, n_periods=3, n_splits=2))

        # A dev checkout without package metadata reports "unknown", which is not a version.
        expected = None if chap_core.__version__ == "unknown" else chap_core.__version__
        assert backtest.chap_version == expected


def test_every_backtest_parameter_is_part_of_the_uniqueness_key():
    """A parameter that is not in the key would let two incomparable setups share a row.

    Adding a field to BacktestParams therefore widens this constraint on its own, but
    the database still needs a migration to widen it there too.
    """
    from chap_core.api_types import BacktestParams
    from chap_core.database.tables import BacktestSpecification

    constraint = next(
        c
        for c in BacktestSpecification.__table__.constraints
        if getattr(c, "name", None) == "uq_backtestspecification_params"
    )
    assert {column.name for column in constraint.columns} == {
        "dataset_id",
        "target_column",
        *BacktestParams.model_fields,
    }


def test_differing_parameters_produce_different_specifications(p_seeded_engine):
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        two_splits = session.session.get(Backtest, _run(session, "two", dataset_id, n_periods=3, n_splits=2))
        three_splits = session.session.get(Backtest, _run(session, "three", dataset_id, n_periods=3, n_splits=3))

        assert two_splits.specification_id != three_splits.specification_id
        assert (two_splits.n_splits, three_splits.n_splits) == (2, 3)


def test_target_column_is_scored_and_separates_specifications(override_session, p_seeded_engine):
    from chap_core.assessment.evaluation import Evaluation

    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        for observation in session.session.exec(select(Observation).where(Observation.feature_name == "rainfall")):
            observation.value = 7.0
        session.session.commit()
        params = {"n_periods": 3, "n_splits": 2}
        default = session.session.get(Backtest, _run(session, "default", dataset_id, **params))
        rainfall = session.session.get(Backtest, _run(session, "rain", dataset_id, target_column="rainfall", **params))

        assert default.specification_id != rainfall.specification_id
        assert set(Evaluation.from_backtest(rainfall).to_flat().observations.disease_cases) == {7.0}
    rows = client.get("/v1/crud/backtest-specifications", params={"targetColumn": "rainfall"}).json()
    assert [row["id"] for row in rows] == [rainfall.specification_id]
    actual_cases = DataList.model_validate(client.get(f"/v1/analytics/actual-cases/{rainfall.id}").json())
    assert {entry.value for entry in actual_cases.data} == {7.0}
    assert actual_cases.featureId == actual_cases.dhis2Id == "rainfall"


@pytest.mark.parametrize("endpoint", ["create-backtest", "create-backtests"])
def test_create_backtest_rejects_a_mask_in_the_evaluation_window(
    override_session, seeded_session, monkeypatch, endpoint
):
    from chap_core.rest_api.v1.routers import analytics

    worker = _JobIdWorker()
    monkeypatch.setattr(analytics, "worker", worker)
    dataset_id = mask_seeded_dataset(seeded_session, "2022-11", "2022-12")
    payload = {"name": "x", "datasetId": dataset_id, "modelId": "naive_model", "modelIds": ["naive_model"]}

    response = client.post(f"/v1/analytics/{endpoint}", json={**payload, "nPeriods": 3, "nSplits": 2, "stride": 1})
    assert response.status_code == 422, response.text
    assert "falls in the evaluation window" in response.text
    assert worker.queued == []


@pytest.mark.parametrize("endpoint", ["create-backtest", "create-backtests"])
def test_create_backtest_accepts_a_mask_in_the_history(override_session, seeded_session, monkeypatch, endpoint):
    from chap_core.rest_api.v1.routers import analytics

    worker = _JobIdWorker()
    monkeypatch.setattr(analytics, "worker", worker)
    dataset_id = mask_seeded_dataset(seeded_session, "2021-06", "2021-08")
    payload = {"name": "x", "datasetId": dataset_id, "modelId": "naive_model", "modelIds": ["naive_model"]}

    response = client.post(f"/v1/analytics/{endpoint}", json={**payload, "nPeriods": 3, "nSplits": 2, "stride": 1})
    assert response.status_code == 200, response.text
    assert len(worker.queued) == 1


def test_run_backtest_masks_the_history_without_writing_to_the_dataset(
    override_session, p_seeded_engine, seeded_session
):
    dataset_id = mask_seeded_dataset(seeded_session, "2021-06", "2021-08")
    with SessionWrapper(p_seeded_engine) as session:
        before = DataSetManager(session.session).to_dataset(dataset_id)
        model = session.get_configured_model_by_id_or_name("naive_model")
        backtest_id = run_backtest(
            BacktestCreate(name="masked", dataset_id=dataset_id, model_id=model.id),
            n_periods=3,
            n_splits=2,
            stride=1,
            session=session,
        )
        assert session.session.get(Backtest, backtest_id) is not None
        after = DataSetManager(session.session).to_dataset(dataset_id)
        for location in before.keys():
            assert np.array_equal(before[location].disease_cases, after[location].disease_cases, equal_nan=True)


def test_run_backtest_rejects_a_mask_in_the_evaluation_window(override_session, p_seeded_engine, seeded_session):
    dataset_id = mask_seeded_dataset(seeded_session, "2022-11", "2022-12")
    with SessionWrapper(p_seeded_engine) as session:
        model = session.get_configured_model_by_id_or_name("naive_model")
        with pytest.raises(ValueError, match="falls in the evaluation window"):
            run_backtest(
                BacktestCreate(name="masked", dataset_id=dataset_id, model_id=model.id),
                n_periods=3,
                n_splits=2,
                stride=1,
                session=session,
            )


def test_differing_weather_providers_produce_different_specifications(p_seeded_engine):
    """Results scored against different climate covariates are not comparable, so the
    provider is part of the key just like the split parameters are."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        params = {"n_periods": 3, "n_splits": 2, "stride": 1, "n_retrain": 1}
        climatology = session.session.get(
            Backtest, _run(session, "climatology", dataset_id, future_weather_provider="climatology", **params)
        )
        observed = session.session.get(
            Backtest, _run(session, "observed", dataset_id, future_weather_provider="observed", **params)
        )

        assert climatology.specification_id != observed.specification_id
        assert observed.specification.future_weather_provider == "observed"


def test_differing_datasets_produce_different_specifications(p_seeded_engine):
    """The dataset is part of the key: identical parameters over different data are not comparable."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_ids = session.session.exec(select(DataSet.id).order_by(DataSet.id)).all()
        params = {"n_periods": 3, "n_splits": 2, "stride": 1, "n_retrain": 1}
        specification_ids = {
            session.session.get(Backtest, _run(session, f"ds{dataset_id}", dataset_id, **params)).specification_id
            for dataset_id in dataset_ids[:2]
        }

        assert len(specification_ids) == 2


def test_specification_org_units_are_the_ones_left_after_filtering(p_seeded_engine):
    """org_units belongs on the specification because the filter that produces it reads
    the parameters: a longer evaluation window shortens the training window and can drop
    an org unit whose only target values sit in the part that is now evaluated."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id).where(DataSet.name == "dataset_with_nans")).first()
        dataset = DataSetManager(session.session).to_dataset(dataset_id)
        backtest = session.session.get(Backtest, _run(session, "filtered", dataset_id, n_periods=2, n_splits=2))

        assert set(backtest.specification.org_units) < set(dataset.locations())
        assert set(backtest.specification.org_units) == set(backtest.org_units)


def test_specification_filter_covers_every_backtest_parameter():
    """A parameter missing from the filter could not be part of the lookup tuple, so an
    external system could not find its specification again. Every field must also be
    optional with no default: an omitted parameter means any value, not the default."""
    from chap_core.api_types import BacktestParams
    from chap_core.rest_api.data_models import BacktestSpecificationFilter

    assert set(BacktestSpecificationFilter.model_fields) == {
        "dataset_id",
        "target_column",
        *BacktestParams.model_fields,
    }
    assert all(field.default is None for field in BacktestSpecificationFilter.model_fields.values())


def test_list_backtest_specifications_filters_by_dataset_and_parameters(
    override_session, p_seeded_engine, backtest_params, org_units
):
    """The seeded database holds two specifications with the same parameters on two datasets."""
    response = client.get("/v1/crud/backtest-specifications")
    assert response.status_code == 200, response.text
    rows = response.json()
    assert len(rows) == 2, rows
    assert len({row["dataset"]["id"] for row in rows}) == 2, rows
    assert all(row["backtestCount"] == 1 and row["orgUnitCount"] == len(org_units) for row in rows), rows
    assert all("backtests" not in row for row in rows)

    by_dataset = client.get("/v1/crud/backtest-specifications", params={"datasetId": rows[0]["dataset"]["id"]}).json()
    assert [row["id"] for row in by_dataset] == [rows[0]["id"]]

    full_tuple = {"datasetId": rows[0]["dataset"]["id"], **backtest_params.model_dump(by_alias=True)}
    assert [row["id"] for row in client.get("/v1/crud/backtest-specifications", params=full_tuple).json()] == [
        rows[0]["id"]
    ]
    assert client.get("/v1/crud/backtest-specifications", params={**full_tuple, "nSplits": 99}).json() == []


def test_get_backtest_specification_returns_its_backtests_newest_first(override_session, p_seeded_engine):
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        params = {"n_periods": 3, "n_splits": 2, "stride": 1, "n_retrain": 1}
        first_id = _run(session, "first", dataset_id, **params)
        second_id = _run(session, "second", dataset_id, **params)
        specification_id = session.session.get(Backtest, first_id).specification_id
        # The seeded backtest has the same parameters, so it sits on the same
        # specification; it predates `created`, which must not push it to the top.
        seeded_id = session.session.exec(select(Backtest.id).where(Backtest.name == "test backtest")).one()

    response = client.get(f"/v1/crud/backtest-specifications/{specification_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["id"] == specification_id
    assert data["dataset"]["id"] == dataset_id
    assert (data["nPeriods"], data["nSplits"]) == (3, 2)
    assert data["orgUnits"]
    assert [b["id"] for b in data["backtests"]] == [second_id, first_id, seeded_id]
    for row in data["backtests"]:
        BacktestRead.model_validate(row)
        assert row["specificationId"] == specification_id
        assert row["configuredModel"]["configurationDigest"]
        assert row["configuredModel"]["modelTemplate"]["name"]
        assert isinstance(row["aggregateMetrics"], dict)
        assert "forecasts" not in row

    assert client.get("/v1/crud/backtest-specifications/999999").status_code == 404


def test_list_backtests_filters_by_specification_and_dataset(override_session, p_seeded_engine):
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        params = {"n_periods": 3, "n_splits": 2, "stride": 1, "n_retrain": 1}
        first_id = _run(session, "first", dataset_id, **params)
        second_id = _run(session, "second", dataset_id, **params)
        other_id = _run(session, "other", dataset_id, **{**params, "n_splits": 3})
        specification_id = session.session.get(Backtest, first_id).specification_id

    by_specification = client.get("/v1/crud/backtests", params={"specificationId": specification_id}).json()
    assert {first_id, second_id} <= {b["id"] for b in by_specification}
    assert other_id not in {b["id"] for b in by_specification}
    assert all(b["specificationId"] == specification_id for b in by_specification)

    by_dataset = client.get("/v1/crud/backtests", params={"datasetId": dataset_id}).json()
    assert {first_id, second_id, other_id} <= {b["id"] for b in by_dataset}
    assert all(b["datasetId"] == dataset_id for b in by_dataset)


def test_backtest_listings_hide_baseline_backtests(override_session, p_seeded_engine):
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        params = {"n_periods": 3, "n_splits": 2, "stride": 1, "n_retrain": 1}
        baseline_id = _run(session, "baseline", dataset_id, **params)
        # Same regions and splits, so without the filter it would list the baseline as compatible.
        other_id = _run(session, "other", dataset_id, **params)
        backtest = session.session.get(Backtest, baseline_id)
        specification_id = backtest.specification_id
        template = backtest.configured_model.model_template
        template.role = ModelTemplateRole.baseline
        session.session.add(template)
        session.session.commit()

    # The backtest is still there, and it carries the role of its template.
    response = client.get(f"/v1/crud/backtests/{baseline_id}")
    assert response.status_code == 200, response.text
    assert response.json()["configuredModel"]["modelTemplate"]["role"] == "baseline"

    assert baseline_id not in {b["id"] for b in client.get("/v1/crud/backtests").json()}
    specification = client.get(f"/v1/crud/backtest-specifications/{specification_id}").json()
    assert baseline_id not in {b["id"] for b in specification["backtests"]}
    summaries = client.get("/v1/crud/backtest-specifications").json()
    counts = {row["id"]: row["backtestCount"] for row in summaries}
    assert counts.get(specification_id, 0) == len(specification["backtests"])
    compatible = client.get(f"/v1/analytics/compatible-backtests/{other_id}").json()
    assert baseline_id not in {b["id"] for b in compatible}


def test_backtest_read_still_exposes_the_parameters_flat(override_session, p_seeded_engine):
    """The parameters moved behind a relationship; the wire shape must not have moved with them."""
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest_id = _run(session, "read", dataset_id, n_periods=3, n_splits=2, stride=1, n_retrain=2)

    response = client.get(f"/v1/crud/backtests/{backtest_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["nPeriods"], body["nSplits"], body["stride"], body["nRetrain"]) == (3, 2, 1, 2)
    # Lock the generated contract down: a later refactor must not quietly add or drop a
    # field the frontend reads.
    assert set(BacktestRead.model_json_schema(mode="serialization")["properties"]) == {
        "nPeriods",
        "nSplits",
        "stride",
        "nRetrain",
        "futureWeatherProvider",
        "datasetId",
        "modelId",
        "name",
        "created",
        "modelTemplateVersion",
        "id",
        "orgUnits",
        "splitPeriods",
        "maxHorizonDistance",
        "chapVersion",
        "dataset",
        "specificationId",
        "aggregateMetrics",
        "configuredModel",
        "predictionSetupId",
    }
    assert set(body) == set(BacktestRead.model_json_schema(mode="serialization")["properties"])

    # /full serialises the table class itself, so its parameters resolve through a lazy
    # load that has to happen while the request session is still open.
    full = client.get(f"/v1/crud/backtests/{backtest_id}/full")
    assert full.status_code == 200, full.text
    assert (full.json()["nPeriods"], full.json()["nSplits"]) == (3, 2)


def test_run_backtest_retrains_n_retrain_times(p_seeded_engine, monkeypatch):
    """n_retrain must reach the evaluator, not just the row."""
    from chap_core.predictor.naive_estimator import NaiveEstimator

    class _CountingEstimator:
        def __init__(self):
            self.inner = NaiveEstimator()
            self.train_calls = 0

        def train(self, data):
            self.train_calls += 1
            return self.inner.train(data)

    estimator = _CountingEstimator()
    monkeypatch.setattr(SessionWrapper, "get_configured_model_with_code", lambda self, *args, **kwargs: estimator)

    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest_id = run_backtest(
            BacktestCreate(name="retrain", dataset_id=dataset_id, model_id="naive_model"),
            n_periods=3,
            n_splits=4,
            stride=1,
            n_retrain=2,
            session=session,
        )
        assert estimator.train_calls == 2
        assert session.session.get(Backtest, backtest_id).n_retrain == 2


def test_run_backtest_rejects_model_that_skips_an_org_unit(p_seeded_engine, monkeypatch, org_units):
    """A model that forecasts fewer org units than it was given must fail, not be stored."""
    from chap_core.assessment.prediction_evaluator import IncompleteBacktestError
    from chap_core.predictor.naive_estimator import NaiveEstimator
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as SpatioTemporalDataSet

    dropped = org_units[0]

    class _DroppingPredictor:
        def __init__(self, inner):
            self.inner = inner

        def predict(self, historic_data, future_data):
            forecasts = self.inner.predict(historic_data, future_data)
            return SpatioTemporalDataSet({loc: s for loc, s in forecasts.items() if loc != dropped})

    class _DroppingEstimator:
        def train(self, data):
            return _DroppingPredictor(NaiveEstimator().train(data))

    monkeypatch.setattr(
        SessionWrapper, "get_configured_model_with_code", lambda self, *args, **kwargs: _DroppingEstimator()
    )

    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        n_backtests_before = len(session.session.exec(select(Backtest)).all())
        with pytest.raises(IncompleteBacktestError) as excinfo:
            run_backtest(
                BacktestCreate(name="incomplete", dataset_id=dataset_id, model_id="naive_model"),
                n_periods=3,
                n_splits=2,
                stride=1,
                session=session,
            )
        assert dropped in str(excinfo.value)
        assert len(session.session.exec(select(Backtest)).all()) == n_backtests_before


def test_backtest_overlap_error_message_includes_id(clean_engine, dependency_overrides):
    """The error detail must surface the actual id, not its path position."""
    missing_id1 = 888888
    missing_id2 = 999999
    response = client.get(f"/v1/analytics/backtest-overlap/{missing_id1}/{missing_id2}")
    assert response.status_code == 404, response.text
    assert str(missing_id1) in response.json()["detail"], response.json()


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
    check_all_regions_rejected_dry_run(
        "/v1/analytics/create-backtest-with-data", create_backtest_with_data_request.model_dump()
    )


def test_backtest_with_no_target_data_in_any_region_returns_400_with_rejections(
    dependency_overrides, create_backtest_with_data_request
):
    request_payload = create_backtest_with_data_request.model_dump()
    for observation in request_payload["provided_data"]:
        if observation["feature_name"] == "disease_cases":
            observation["value"] = None
    response = client.post("/v1/analytics/create-backtest-with-data", json=request_payload)
    assert response.status_code == 400, response.json()
    detail = response.json()["detail"]
    assert detail["imported_count"] == 0
    assert detail["rejected"]
    assert all(r["featureName"] == "disease_cases" for r in detail["rejected"])


@pytest.mark.parametrize("dry_run", [False, True])
def test_backtest_with_weekly_data_flow(
    celery_session_worker, dependency_overrides, example_polygons, create_backtest_with_weekly_data_request, dry_run
):
    request_payload = create_backtest_with_weekly_data_request.model_dump()
    print(request_payload)
    _check_backtest_with_data(request_payload, expected_rejections=[], dry_run=dry_run, expected_period_type="week")


def _check_backtest_with_data(request_payload, expected_rejections=None, dry_run=False, expected_period_type="month"):
    url = "/v1/analytics/create-backtest-with-data"
    if dry_run:
        url += "?dryRun=true"
    response = client.post(url, json=request_payload)
    content = response.json()
    assert response.status_code == 200, content
    check_rejected_org_units(content, expected_rejections)
    job_id = content["id"]
    if dry_run:
        assert job_id is None, "Job ID should be None for dry run"
        return
    db_id = await_result_id(job_id, timeout=180)
    response = client.get(f"/v1/crud/backtests/{db_id}/info")
    assert response.status_code == 200, response.json()
    backtest_info = BacktestRead.model_validate(response.json())
    assert backtest_info.stride == request_payload["stride"]
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


def test_all_backtest_plots_via_api(override_session, seeded_session):
    """Test that all registered backtest plots can be generated via the REST API."""
    # First, list all available backtest plot types
    response = client.get("/v1/visualization/backtest-plots/")
    assert response.status_code == 200, response.json()
    plot_types = response.json()
    assert len(plot_types) > 0, "No backtest plot types registered"

    # Get a backtest ID from the seeded database
    from sqlmodel import select

    from chap_core.database.tables import Backtest

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
