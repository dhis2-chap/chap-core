import logging

import numpy as np
import pytest
from sqlmodel import Session, select

from chap_core.api_types import BacktestParams, PredictionEntry
from chap_core.assessment.metrics import get_comparison_description
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import (
    DataSet,
    DataSetCreateInfo,
    MaskedRange,
)
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.database.settings import SettingsManager, default_settings
from chap_core.database.tables import (
    Backtest,
    BacktestSpecification,
    Prediction,
    PredictionInfo,
    PredictionSamplesEntry,
    PredictionSetupObservation,
    SpecificationBaseline,
    SpecificationBaselineStatus,
)
from chap_core.datatypes import create_tsdataclass
from chap_core.rest_api.data_models import (
    MakePredictionRequest,
    PredictionParams,
)
from chap_core.rest_api.db_worker_functions import (
    predict_pipeline_from_composite_dataset,
    run_prediction,
)
from chap_core.spatio_temporal_data.converters import observations_to_dataset
from chap_core.time_period import TimePeriod
from tests.integration.rest_api.db_endpoint_helpers import (
    await_result_id,
    client,
    create_make_data_request,
    create_prediction_setup,
    mask_seeded_dataset,
    prediction_setup_payload,
)


def test_run_prediction_trains_on_a_masked_dataset(override_session, p_seeded_engine, seeded_session, monkeypatch):
    from chap_core.rest_api import db_worker_functions

    seen = []
    forecast_ahead = db_worker_functions.forecast_ahead

    def recording_forecast_ahead(estimator, dataset, *args, **kwargs):
        seen.append(dataset)
        return forecast_ahead(estimator, dataset, *args, **kwargs)

    monkeypatch.setattr(db_worker_functions, "forecast_ahead", recording_forecast_ahead)
    dataset_id = mask_seeded_dataset(seeded_session, "2021-06", "2021-08")
    with SessionWrapper(p_seeded_engine) as session:
        prediction_id = run_prediction("naive_model", str(dataset_id), 3, "masked", session=session)
        assert session.session.get(Prediction, prediction_id) is not None
    trained_on = seen[0]
    masked = [
        period.id
        for period, flag in zip(trained_on.period_range, next(iter(trained_on.values())).target_masked)
        if flag
    ]
    assert masked == ["202106", "202107", "202108"]
    assert all(np.all(np.isnan(data.disease_cases[17:20])) for data in trained_on.values())


def test_create_prediction_setup_happy_path(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    response = create_prediction_setup(backtest.id, "Setup A")
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

    response = create_prediction_setup(backtest.id, "Snapshot check")
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
        json=prediction_setup_payload(99999, "ghost"),
    )
    assert response.status_code == 404


def test_create_prediction_setup_invalid_cron_returns_422(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    payload = prediction_setup_payload(backtest.id, "Bad cron")
    payload["scheduleCronExpression"] = "not a cron expression"
    response = client.post("/v1/crud/prediction-setups", json=payload)
    assert response.status_code == 422


def test_create_prediction_setup_enabled_without_expression_returns_422(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    payload = prediction_setup_payload(backtest.id, "Enabled but empty")
    payload["scheduleCronExpression"] = None
    payload["scheduleEnabled"] = True
    response = client.post("/v1/crud/prediction-setups", json=payload)
    assert response.status_code == 422


def test_create_prediction_setup_duplicate_returns_409(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    first = create_prediction_setup(backtest.id, "First")
    assert first.status_code == 200, first.json()
    second = create_prediction_setup(backtest.id, "Second")
    assert second.status_code == 409


def test_list_prediction_setups_returns_created_setup(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Listed setup")
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
    created = create_prediction_setup(backtest.id, "With predictions")
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
    assert body["predictions"][0]["predictionSetupId"] == setup_id


def test_prediction_setup_monitoring_compares_with_the_default_baseline(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Monitored")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]
    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None
    prediction.prediction_setup_id = setup_id
    seeded_session.add(prediction)
    # Stand-in for a later run uploading the observed cases for the prediction's periods.
    for entry in prediction.forecasts:
        seeded_session.add(
            PredictionSetupObservation(
                prediction_setup_id=setup_id, org_unit=entry.org_unit, period=entry.period, disease_cases=1.0
            )
        )
    backtest.aggregate_metrics = {"mae": 1.5}
    # Stand-ins for the baseline's runs that forecast exactly what the model did, so every comparison is 0.
    baseline = SessionWrapper(session=seeded_session).get_configured_model_by_id_or_name("seasonal_shift_persistence")
    seeded_session.add(
        SpecificationBaseline(
            specification_id=backtest.specification_id,
            configured_model_id=baseline.id,
            status=SpecificationBaselineStatus.done,
            backtest_id=backtest.id,
        )
    )
    seeded_session.add(
        Prediction(
            **prediction.model_dump(
                exclude={"id", "prediction_setup_id", "model_id", "model_db_id", "baseline_for_prediction_id"}
            ),
            model_id=baseline.name,
            model_db_id=baseline.id,
            baseline_for_prediction_id=prediction.id,
            forecasts=[
                PredictionSamplesEntry(period=entry.period, org_unit=entry.org_unit, values=entry.values)
                for entry in prediction.forecasts
            ],
        )
    )
    seeded_session.commit()

    response = client.get(f"/v1/crud/prediction-setups/{setup_id}/monitoring", params={"metric": "mae"})

    assert response.status_code == 200, response.json()
    body = response.json()
    assert (body["metricId"], body["baselineModel"]) == ("mae", "seasonal_shift_persistence")
    assert body["comparisonDescription"] == get_comparison_description("skill_ratio")
    assert body["evaluationValue"] == 0.0
    sources = [point["source"] for point in body["points"]]
    # Evaluation points come first, one per backtest split, then one per prediction.
    assert sources == sorted(sources)
    assert sources.count("evaluation") == len({forecast.last_seen_period for forecast in backtest.forecasts})
    assert {point["value"] for point in body["points"]} == {0.0}
    first_forecast_period = TimePeriod.parse(min(entry.period for entry in prediction.forecasts))
    [prediction_point] = [point for point in body["points"] if point["source"] == "prediction"]
    assert prediction_point["period"] == (first_forecast_period - first_forecast_period.time_delta).id
    assert prediction_point["nObserved"] == len(prediction.forecasts)


@pytest.mark.parametrize("baseline", ["seasonal_shift_persistence", "global_median"])
def test_prediction_setup_monitoring_has_no_points_before_the_baseline_has_run(
    override_session, seeded_session, baseline
):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Monitored")
    assert created.status_code == 200, created.json()

    response = client.get(
        f"/v1/crud/prediction-setups/{created.json()['id']}/monitoring",
        params={"metric": "mae", "baseline": baseline},
    )

    assert response.status_code == 200, response.json()
    body = response.json()
    assert (body["baselineModel"], body["evaluationValue"], body["points"]) == (baseline, None, [])


def test_prediction_setup_monitoring_unknown_baseline_returns_422(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Monitored")
    assert created.status_code == 200, created.json()

    response = client.get(
        f"/v1/crud/prediction-setups/{created.json()['id']}/monitoring",
        params={"metric": "mae", "baseline": "naive_model"},
    )

    assert response.status_code == 422


def test_flat_prediction_forecasts_counts_horizons_from_first_forecast_period(seeded_session):
    from chap_core.services.prediction_monitoring_service import flat_prediction_forecasts

    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None

    forecasts = flat_prediction_forecasts([prediction])

    forecast_periods = sorted({entry.period for entry in prediction.forecasts})
    assert sorted(forecasts.horizon_distance.unique()) == list(range(1, len(forecast_periods) + 1))
    assert set(forecasts[forecasts.horizon_distance == 1].time_period) == {forecast_periods[0]}


def test_prediction_setup_monitoring_unknown_metric_returns_422(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Monitored")
    assert created.status_code == 200, created.json()

    response = client.get(
        f"/v1/crud/prediction-setups/{created.json()['id']}/monitoring", params={"metric": "not_a_metric"}
    )

    assert response.status_code == 422


def test_prediction_setup_monitoring_not_found_returns_404(clean_engine, dependency_overrides):
    response = client.get("/v1/crud/prediction-setups/99999/monitoring", params={"metric": "mae"})
    assert response.status_code == 404


@pytest.mark.parametrize("list_predictions", [False, True])
def test_prediction_response_exposes_setup_id(override_session, seeded_session, list_predictions):
    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None
    endpoint = "/v1/crud/predictions" if list_predictions else f"/v1/crud/predictions/{prediction.id}"

    def read_prediction():
        response = client.get(endpoint)
        assert response.status_code == 200, response.json()
        body = response.json()
        return next(item for item in body if item["id"] == prediction.id) if list_predictions else body

    assert read_prediction()["predictionSetupId"] is None

    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id)
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]
    prediction.prediction_setup_id = setup_id
    seeded_session.add(prediction)
    seeded_session.commit()

    assert read_prediction()["predictionSetupId"] == setup_id


def test_get_prediction_setup_not_found_returns_404(clean_engine, dependency_overrides):
    response = client.get("/v1/crud/prediction-setups/99999")
    assert response.status_code == 404


def test_patch_prediction_setup_updates_name_only(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Original")
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
    created = create_prediction_setup(backtest.id, "Multi-field")
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
    created = create_prediction_setup(backtest.id, "Clearable")
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
    created = create_prediction_setup(backtest.id, "Immutable")
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

    def delete(self, *_keys):
        return 0


def test_delete_prediction_setup_removes_setup_and_keeps_predictions(override_session, seeded_session, monkeypatch):
    from chap_core.rest_api.v1.routers import crud

    monkeypatch.setattr(crud, "redis", _NoopRedis())

    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Deletable")
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
    created = create_prediction_setup(backtest.id, "Empty setup")
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
    created = create_prediction_setup(backtest.id, "Sweep target")
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

        def delete(self, *keys):
            self.deleted.extend(keys)
            for key in keys:
                self.meta.pop(key, None)
            return 1

    fake_redis = _FakeRedis()
    from chap_core.rest_api.v1.routers import crud

    monkeypatch.setattr(crud, "redis", fake_redis)

    response = client.delete(f"/v1/crud/prediction-setups/{setup_id}")
    assert response.status_code == 200, response.json()

    assert fake_redis.deleted == ["job_meta:job-1", "job_request:job-1"]
    assert "job_meta:job-2" in fake_redis.meta


def test_backtest_info_exposes_prediction_setup_id_when_setup_exists(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None

    response = client.get(f"/v1/crud/backtests/{backtest.id}/info")
    assert response.status_code == 200, response.json()
    assert response.json()["predictionSetupId"] is None

    created = create_prediction_setup(backtest.id, "Linked setup")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.get(f"/v1/crud/backtests/{backtest.id}/info")
    assert response.status_code == 200, response.json()
    assert response.json()["predictionSetupId"] == setup_id


def test_backtest_list_exposes_prediction_setup_id(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Listed-link setup")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    response = client.get("/v1/crud/backtests")
    assert response.status_code == 200, response.json()
    matching = next(item for item in response.json() if item["id"] == backtest.id)
    assert matching["predictionSetupId"] == setup_id


def test_run_prediction_setup_not_found_returns_404(clean_engine, dependency_overrides, example_polygons):
    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    # /run body doesn't accept dataToBeFetched / dataSources / maskedRanges — strip them.
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
    payload["nPeriods"] = 3
    response = client.post("/v1/crud/prediction-setups/99999/run", json=payload)
    assert response.status_code == 404, response.json()


def test_run_prediction_setup_rejects_legacy_fields(override_session, seeded_session, example_polygons):
    """The new /run body must reject legacy fields (dataSources, dataToBeFetched,
    configuredModelWithDataSourceId) so old clients fail loud."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Legacy reject")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    # Body still contains dataToBeFetched + dataSources from DatasetMakeRequest — exactly the legacy fields.
    payload["nPeriods"] = 3
    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 422


def _point_at_provider(session, backtest, provider):
    """Move a backtest onto a specification differing only in the weather provider.

    Specifications are immutable and shared between backtests, so changing the provider
    means pointing at a different one rather than editing the one already there.
    """
    specification = BacktestSpecification(
        **backtest.specification.model_dump(exclude={"id"}) | {"future_weather_provider": provider}
    )
    session.add(specification)
    backtest.specification = specification
    session.add(backtest)
    session.commit()


def test_run_prediction_setup_inherits_the_backtest_provider(
    override_session, seeded_session, example_polygons, monkeypatch
):
    """A promoted backtest must predict with the provider it was evaluated on, or the
    prediction silently uses a different future-weather source than the scores imply."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    _point_at_provider(seeded_session, backtest, "damped_persistence")
    setup_id = create_prediction_setup(backtest.id, "Inherit provider").json()["id"]

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
    payload["nPeriods"] = 3

    from chap_core.rest_api.v1.routers import crud as crud_router

    captured: dict = {}

    class _FakeJob:
        id = "captured-job"

    class _CapturingWorker:
        def queue_db(self, func, *args, **kwargs):
            captured.update(kwargs)
            return _FakeJob()

    monkeypatch.setattr(crud_router, "worker", _CapturingWorker())
    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)

    assert response.status_code == 200, response.json()
    assert captured["prediction_params"].future_weather_provider == "damped_persistence"


def _run_setup_capturing_the_job(setup_id, example_polygons, monkeypatch, n_periods=3):
    """Post a monthly 2020-01..2023-12 run to a prediction setup and return the response and the queued kwargs."""
    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(
        mode="json",
        exclude={"data_to_be_fetched", "data_sources", "masked_ranges", "target_column", "provided_data_columns"},
    )
    if n_periods is not None:
        payload["nPeriods"] = n_periods

    from chap_core.rest_api.v1.routers import crud as crud_router

    captured: dict = {}

    class _FakeJob:
        id = "captured-job"

    class _CapturingWorker:
        def queue_db(self, func, *args, **kwargs):
            captured.update(kwargs)
            return _FakeJob()

    monkeypatch.setattr(crud_router, "worker", _CapturingWorker())
    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    return response, captured


@pytest.mark.parametrize(
    "requested, min_periods, max_periods, expected",
    [
        (None, None, None, 7),  # The backtest's horizon.
        (None, None, 2, 2),  # Too long for the model.
        (None, 9, None, 9),  # Too short for the model.
        (4, None, 2, 4),  # An explicit horizon is used as given.
    ],
)
def test_run_prediction_setup_horizon(
    override_session, seeded_session, example_polygons, monkeypatch, requested, min_periods, max_periods, expected
):
    backtest = seeded_session.exec(select(Backtest)).first()
    backtest.specification.n_periods = 7
    template = backtest.configured_model.model_template
    template.min_prediction_periods = min_periods
    template.max_prediction_periods = max_periods
    seeded_session.add_all([backtest.specification, template])
    seeded_session.commit()
    setup_id = create_prediction_setup(backtest.id, "Horizon").json()["id"]

    response, captured = _run_setup_capturing_the_job(setup_id, example_polygons, monkeypatch, n_periods=requested)

    assert response.status_code == 200, response.json()
    assert captured["prediction_params"].n_periods == expected


_WEEKS = [f"2020W{week:02d}" for week in range(1, 53)]


@pytest.mark.parametrize(
    "periods, requested, min_periods, max_periods, expected",
    [
        (None, None, None, None, 5),  # The stored monthly default.
        (_WEEKS, None, None, None, 12),  # The stored weekly default.
        (_WEEKS, None, None, 3, 3),  # Too long for the model.
        (None, None, 9, None, 9),  # Too short for the model.
        (None, 4, None, 2, 4),  # An explicit horizon is used as given.
    ],
)
def test_make_prediction_horizon(
    override_session,
    seeded_session,
    example_polygons,
    baseline_worker,
    monkeypatch,
    periods,
    requested,
    min_periods,
    max_periods,
    expected,
):
    from chap_core.rest_api.v1.routers import analytics

    monkeypatch.setattr(analytics, "worker", baseline_worker)
    model = seeded_session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.name == "naive_model")).one()
    model.model_template.min_prediction_periods = min_periods
    model.model_template.max_prediction_periods = max_periods
    seeded_session.add(model.model_template)
    seeded_session.commit()
    defaults = default_settings()
    SettingsManager(seeded_session).update(
        defaults.model_copy(update={"monthly_backtest_params": BacktestParams(n_periods=5)})
    )
    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"], periods)
    payload = request.model_dump(mode="json", exclude={"data_sources", "masked_ranges"})
    payload["modelId"] = "naive_model"
    if requested is not None:
        payload["nPeriods"] = requested

    response = client.post("/v1/analytics/make-prediction", json=payload)

    assert response.status_code == 200, response.text
    assert baseline_worker.queued[-1][2]["prediction_params"].n_periods == expected


def test_run_prediction_setup_copies_the_backtest_dataset_masked_ranges(
    override_session, seeded_session, example_polygons, monkeypatch
):
    """A promoted backtest must predict under the mask it was scored with: the run's new dataset gets its ranges."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    backtest.dataset.masked_ranges = [MaskedRange(start="2021-06", end="2021-08", name="covid")]
    seeded_session.add(backtest.dataset)
    seeded_session.commit()
    setup_id = create_prediction_setup(backtest.id, "Inherit mask").json()["id"]

    response, captured = _run_setup_capturing_the_job(setup_id, example_polygons, monkeypatch)

    assert response.status_code == 200, response.json()
    assert captured["dataset_create_info"]["masked_ranges"] == [{"start": "202106", "end": "202108", "name": "covid"}]


def test_run_prediction_setup_without_backtest_ranges_creates_an_unmasked_dataset(
    override_session, seeded_session, example_polygons, monkeypatch
):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    setup_id = create_prediction_setup(backtest.id, "No mask").json()["id"]

    response, captured = _run_setup_capturing_the_job(setup_id, example_polygons, monkeypatch)

    assert response.status_code == 200, response.json()
    assert captured["dataset_create_info"]["masked_ranges"] == []


def test_run_prediction_setup_rejects_input_carrying_the_mask_indicator(
    override_session, seeded_session, example_polygons, monkeypatch
):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    setup_id = create_prediction_setup(backtest.id, "Indicator input").json()["id"]
    request = create_make_data_request(
        example_polygons, [], ["rainfall", "disease_cases", "population", "target_masked"]
    )
    payload = request.model_dump(
        mode="json",
        exclude={"data_to_be_fetched", "data_sources", "masked_ranges", "target_column", "provided_data_columns"},
    )
    payload["nPeriods"] = 3

    from chap_core.rest_api.v1.routers import crud as crud_router

    queued: list = []

    class _RecordingWorker:
        def queue_db(self, func, *args, **kwargs):
            queued.append(func)

    monkeypatch.setattr(crud_router, "worker", _RecordingWorker())
    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 422, response.text
    assert "is reserved" in response.text
    assert queued == []


def test_run_prediction_setup_rejects_data_of_another_period_type_than_the_ranges(
    override_session, seeded_session, example_polygons, monkeypatch
):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    backtest.dataset.masked_ranges = [MaskedRange(start="2021W1", end="2021W3")]
    seeded_session.add(backtest.dataset)
    seeded_session.commit()
    setup_id = create_prediction_setup(backtest.id, "Week mask").json()["id"]

    response, captured = _run_setup_capturing_the_job(setup_id, example_polygons, monkeypatch)

    assert response.status_code == 422, response.text
    assert "is a week range, but the dataset has month periods" in response.text
    assert captured == {}


def test_prediction_pipeline_stores_the_ranges_and_trains_masked(
    override_session, p_seeded_engine, example_polygons, monkeypatch
):
    """What a setup run queues: the new prediction dataset keeps the ranges and the model trains on masked data."""
    from chap_core.rest_api import db_worker_functions

    seen = []
    forecast_ahead = db_worker_functions.forecast_ahead

    def recording_forecast_ahead(estimator, dataset, *args, **kwargs):
        seen.append(dataset)
        return forecast_ahead(estimator, dataset, *args, **kwargs)

    monkeypatch.setattr(db_worker_functions, "forecast_ahead", recording_forecast_ahead)
    features = ["rainfall", "disease_cases", "population"]
    request = create_make_data_request(example_polygons, [], features)
    provided = observations_to_dataset(create_tsdataclass(features), request.provided_data, fill_missing=True)
    provided.set_polygons(request.geojson)
    ranges = [MaskedRange(start="2021-06", end="2021-08", name="covid")]

    with SessionWrapper(p_seeded_engine) as session:
        prediction_id = predict_pipeline_from_composite_dataset(
            features,
            provided.model_dump(),
            "masked-run",
            dataset_create_info=DataSetCreateInfo(name="masked-run", type="prediction", masked_ranges=ranges),
            prediction_params=PredictionParams(model_id="naive_model", n_periods=3),
            session=session,
        )
        prediction = session.session.get(Prediction, prediction_id)
        assert session.session.get(DataSet, prediction.dataset_id).masked_ranges == ranges

    trained_on = seen[0]
    flags = next(iter(trained_on.values())).target_masked
    assert [period.id for period, flag in zip(trained_on.period_range, flags) if flag] == ["202106", "202107", "202108"]


def test_run_prediction_setup_forecasts_the_backtest_target(
    override_session, seeded_session, p_seeded_engine, example_polygons, monkeypatch
):
    from types import SimpleNamespace

    from chap_core.rest_api.celery_tasks import JOB_METADATA_KW, JOB_NAME_KW, JOB_REQUEST_KW, JOB_TYPE_KW
    from chap_core.rest_api.v1.routers import crud

    backtest = seeded_session.exec(select(Backtest)).first()
    backtest.model_db_id = seeded_session.exec(
        select(ConfiguredModelDB.id).where(ConfiguredModelDB.name == "naive_model")
    ).one()
    backtest.specification = BacktestSpecification(
        **backtest.specification.model_dump(exclude={"id"}) | {"target_column": "hospitalisations"}
    )
    seeded_session.commit()
    setup_id = create_prediction_setup(backtest.id, "Target forecast").json()["id"]

    request = create_make_data_request(example_polygons, [], ["rainfall", "hospitalisations", "disease_cases"])
    for observation in request.provided_data:
        observation.value = 0.0 if observation.feature_name == "hospitalisations" else 1000.0
        if observation.feature_name != "rainfall" and observation.period == "2020-01":
            observation.value = None  # Target gaps must not reject otherwise valid locations.
    payload = request.model_dump(
        mode="json",
        exclude={"data_to_be_fetched", "data_sources", "masked_ranges", "target_column", "provided_data_columns"},
    )
    payload["nPeriods"] = 3

    def run_inline(func, *args, **kwargs):
        for key in (JOB_TYPE_KW, JOB_NAME_KW, JOB_REQUEST_KW, JOB_METADATA_KW):
            kwargs.pop(key)
        kwargs["prediction_params"] = kwargs["prediction_params"].model_dump()
        with SessionWrapper(p_seeded_engine) as session:
            prediction_id = func(*args, session=session, **kwargs)
        return SimpleNamespace(id=str(prediction_id))

    monkeypatch.setattr(crud.worker, "queue_db", run_inline)
    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 200, response.text

    prediction = seeded_session.get(Prediction, int(response.json()["id"]))
    assert prediction.dataset.target_column == "hospitalisations"
    assert len(prediction.forecasts) == 3 * len(example_polygons.features)
    # Training on hospitalisations (all zero) rather than disease_cases gives all-zero samples.
    assert all(forecast.values and set(forecast.values) == {0.0} for forecast in prediction.forecasts)


def test_run_prediction_setup_rejects_missing_target(override_session, seeded_session, example_polygons):
    backtest = seeded_session.exec(select(Backtest)).first()
    setup_id = create_prediction_setup(backtest.id, "Missing target").json()["id"]
    request = create_make_data_request(example_polygons, [], ["rainfall"])
    payload = request.model_dump(
        mode="json",
        exclude={"data_to_be_fetched", "data_sources", "masked_ranges", "target_column", "provided_data_columns"},
    )

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 422, response.text


def test_run_prediction_setup_rejects_a_look_ahead_provider(override_session, seeded_session, example_polygons):
    """`observed` reads the forecast window's own weather, so it cannot forecast ahead.
    Fail at the endpoint rather than deep inside the worker."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    _point_at_provider(seeded_session, backtest, "observed")
    setup_id = create_prediction_setup(backtest.id, "Look-ahead setup").json()["id"]

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
    payload["nPeriods"] = 3

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)

    assert response.status_code == 400, response.json()
    assert "cannot forecast ahead" in response.json()["detail"]


@pytest.mark.parametrize("n_periods", [0, -1])
def test_run_prediction_setup_non_positive_n_periods_returns_422(
    override_session, seeded_session, example_polygons, n_periods
):
    """Pydantic gt=0 on RunPredictionSetupRequest.n_periods should reject 0 and negatives
    before the handler runs. Lock in the contract."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, f"n_periods={n_periods}")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
    payload["nPeriods"] = n_periods

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 422, response.json()


def test_run_prediction_setup_empty_provided_data_returns_422(override_session, seeded_session, example_polygons):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Empty data")
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
    created = create_prediction_setup(backtest.id, "Archived target")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    configured_model.archived = True
    seeded_session.add(configured_model)
    seeded_session.commit()

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
    payload["nPeriods"] = 3

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 409, response.json()


def test_run_prediction_setup_queues_the_pinned_configured_model(
    override_session, seeded_session, example_polygons, monkeypatch
):
    """A setup must run the configuration it was created with, not whatever its name now resolves to."""
    from chap_core.rest_api.v1.routers import crud

    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Pinned model")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    captured: dict = {}

    class _FakeJob:
        id = "captured-job"

    class _CapturingWorker:
        def queue_db(self, *args, **kwargs):
            captured.update(kwargs)
            return _FakeJob()

    monkeypatch.setattr(crud, "worker", _CapturingWorker())

    request = create_make_data_request(example_polygons, [], ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
    payload["nPeriods"] = 3

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 200, response.json()
    assert captured["configured_model_id"] == backtest.model_db_id


def test_run_prediction_setup_logs_rejection_count(
    override_session, seeded_session, example_polygons, monkeypatch, caplog
):
    """When validate_full_dataset reports rejections, the router should log a warning
    rather than swallow them silently."""
    from chap_core.rest_api.data_models import ValidationError
    from chap_core.rest_api.v1.routers import crud

    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    created = create_prediction_setup(backtest.id, "Rejection log")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    def fake_validate(_feature_names, dataset, _target_column):
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
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
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
    created = create_prediction_setup(backtest.id, "Redis-down target")
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
    created = create_prediction_setup(backtest.id, "Type normalize")
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
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
    payload["nPeriods"] = 3
    payload["type"] = "forecasting"  # what chap-scheduler currently sends; should be overridden server-side

    response = client.post(f"/v1/crud/prediction-setups/{setup_id}/run", json=payload)
    assert response.status_code == 200, response.json()

    dataset_info = captured["kwargs"]["dataset_create_info"]
    assert dataset_info["type"] == "prediction"
    assert captured["kwargs"]["prediction_setup_id"] == setup_id


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

    created = create_prediction_setup(backtest_id, "End-to-end run")
    assert created.status_code == 200, created.json()
    setup_id = created.json()["id"]

    request = create_make_data_request(example_polygons, [], provided_features)
    payload = request.model_dump(mode="json")
    payload.pop("data_to_be_fetched", None)
    payload.pop("data_sources", None)
    payload.pop("masked_ranges", None)
    payload.pop("target_column", None)
    payload.pop("provided_data_columns", None)
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
