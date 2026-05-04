"""Smoke tests for the /v1/xai router."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel
from starlette.testclient import TestClient

from chap_core.database.tables import Prediction, PredictionSamplesEntry
from chap_core.database.xai_tables import PredictionExplanation
from chap_core.rest_api.app import app
from chap_core.rest_api.v1.routers.dependencies import get_session
from chap_core.rest_api.v1.routers.xai import worker


@pytest.fixture
def engine(tmp_path):
    db_path = tmp_path / "xai_router_test.sqlite"
    eng = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def client(engine):
    def _override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _override
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def prediction_with_cached_global(engine):
    cached_entry = {
        "topFeatures": [
            {"feature_name": "rainfall", "importance": 0.42, "direction": "positive"},
        ],
        "computedAt": datetime.datetime.now(datetime.UTC).isoformat(),
        "nSamples": 12,
        "stabilityScore": 0.9,
        "surrogateQuality": None,
    }
    with Session(engine) as session:
        prediction = Prediction(
            model_id="naive_model",
            model_db_id=1,
            dataset_id=1,
            n_periods=3,
            name="cached prediction",
            created=datetime.datetime.now(),
            meta_data={"xai": {"global_by_method": {"shap_auto": cached_entry}}},
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction.id


def test_list_methods_returns_non_empty(client):
    resp = client.get("/v1/xai/methods")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) > 0
    names = {m["name"] for m in body}
    assert "shap_auto" in names


def test_global_explanation_returns_cached_entry(client, prediction_with_cached_global):
    prediction_id = prediction_with_cached_global
    resp = client.get(f"/v1/xai/predictions/{prediction_id}/global", params={"xaiMethod": "shap_auto"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "shap_auto"
    assert body["nSamples"] == 12
    assert body["stabilityScore"] == 0.9
    assert body["topFeatures"][0]["featureName"] == "rainfall"


@pytest.fixture
def prediction_with_native_shap_local(engine):
    """Prediction with a stored native_shap PredictionExplanation using the canonical
    calendar period format (YYYYMM) as produced by _store_native_shap_explanations."""
    with Session(engine) as session:
        prediction = Prediction(
            model_id="native_model",
            model_db_id=1,
            dataset_id=1,
            n_periods=1,
            name="native shap prediction",
            created=datetime.datetime.now(),
            meta_data={"xai": {"global_by_method": {"native_shap": {"topFeatures": [], "nSamples": 1}}}},
        )
        session.add(prediction)
        session.flush()

        forecast = PredictionSamplesEntry(
            prediction_id=prediction.id,
            period="202406",
            org_unit="OU123",
            values=[10.0],
        )
        session.add(forecast)

        explanation = PredictionExplanation(
            prediction_id=prediction.id,
            org_unit="OU123",
            period="202406",
            method="native_shap",
            output_statistic="median",
            params={},
            result={
                "feature_attributions": [{"feature_name": "rainfall", "importance": 0.5}],
                "baseline_prediction": 8.0,
                "actual_prediction": 10.0,
                "surrogate_quality": None,
                "covariate_provenance": None,
            },
            status="completed",
        )
        session.add(explanation)
        session.commit()
        session.refresh(prediction)
        return prediction.id


def test_native_shap_local_explanation_resolved_by_stored_period(client, prediction_with_native_shap_local):
    """POST /local with xaiMethod=native_shap must find the stored explanation using the
    canonical calendar period even when the request sends step-notation (e.g. '202406_1')."""
    prediction_id = prediction_with_native_shap_local
    resp = client.post(
        f"/v1/xai/predictions/{prediction_id}/local",
        json={"orgUnit": "OU123", "period": "202406_1", "xaiMethod": "native_shap"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "native_shap"
    assert body["period"] == "202406"
    assert body["orgUnit"] == "OU123"


@pytest.fixture
def prediction_with_native_shap_canonical_period(engine):
    """Prediction where the model stored canonical calendar periods in shap_values.csv (e.g. '202407')
    rather than step-notation (e.g. '202406_1'). The POST request still sends step-notation."""
    with Session(engine) as session:
        prediction = Prediction(
            model_id="native_model_canonical",
            model_db_id=1,
            dataset_id=1,
            n_periods=1,
            name="native shap canonical period",
            created=datetime.datetime.now(),
            meta_data={"xai": {"global_by_method": {"native_shap": {"topFeatures": [], "nSamples": 1}}}},
        )
        session.add(prediction)
        session.flush()

        forecast = PredictionSamplesEntry(
            prediction_id=prediction.id,
            period="202407",
            org_unit="OU123",
            values=[10.0],
        )
        session.add(forecast)

        explanation = PredictionExplanation(
            prediction_id=prediction.id,
            org_unit="OU123",
            period="202407",
            method="native_shap",
            output_statistic="median",
            params={},
            result={
                "feature_attributions": [{"feature_name": "rainfall", "importance": 0.5}],
                "baseline_prediction": 8.0,
                "actual_prediction": 10.0,
                "surrogate_quality": None,
                "covariate_provenance": None,
            },
            status="completed",
        )
        session.add(explanation)
        session.commit()
        session.refresh(prediction)
        return prediction.id


def test_native_shap_local_explanation_resolved_by_canonical_period(
    client, prediction_with_native_shap_canonical_period
):
    """POST /local must also find stored native_shap explanations whose period is the canonical
    calendar period (e.g. '202407') when the request sends step-notation (e.g. '202406_1')."""
    prediction_id = prediction_with_native_shap_canonical_period
    resp = client.post(
        f"/v1/xai/predictions/{prediction_id}/local",
        json={"orgUnit": "OU123", "period": "202406_1", "xaiMethod": "native_shap"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "native_shap"
    assert body["orgUnit"] == "OU123"


def test_native_shap_dashed_period_normalized_at_storage(engine, client):
    """_store_native_shap_explanations must normalize 'YYYY-MM' to 'YYYYMM' at storage
    time so that GET/POST /local can find explanations with a plain equality query."""
    from sqlmodel import select

    from chap_core.database.database import SessionWrapper
    from chap_core.rest_api.db_worker_functions import _store_native_shap_explanations

    with Session(engine) as session:
        prediction = Prediction(
            model_id="chtorch_model",
            model_db_id=1,
            dataset_id=1,
            n_periods=1,
            name="native shap dashed period",
            created=datetime.datetime.now(),
            meta_data={"xai": {"global_by_method": {"native_shap": {"topFeatures": [], "nSamples": 1}}}},
        )
        session.add(prediction)
        session.flush()
        forecast = PredictionSamplesEntry(
            prediction_id=prediction.id,
            period="202407",
            org_unit="OU123",
            values=[10.0],
        )
        session.add(forecast)
        session.commit()
        prediction_id = prediction.id

    native_shap = {
        "feature_names": ["rainfall"],
        "values": [
            {
                "time_period": "2024-07",
                "location": "OU123",
                "shap_values": [0.5],
                "feature_values": {"rainfall": 1.5},
                "expected_value": 8.0,
            }
        ],
    }
    with Session(engine) as session:
        _store_native_shap_explanations(native_shap, prediction_id, SessionWrapper(session=session))

    with Session(engine) as session:
        exp = session.exec(
            select(PredictionExplanation).where(PredictionExplanation.prediction_id == prediction_id)
        ).first()
        assert exp is not None
        assert exp.period == "202407"

    get_resp = client.get(
        f"/v1/xai/predictions/{prediction_id}/local",
        params={"orgUnit": "OU123", "period": "202407", "xaiMethod": "native_shap"},
    )
    assert get_resp.status_code == 200, get_resp.text
    assert len(get_resp.json()) == 1


def test_run_explanations_uses_prediction_name_and_method_in_job_name(
    client, prediction_with_cached_global, monkeypatch
):
    captured: dict[str, Any] = {}

    def _fake_queue_db(func, *args, **kwargs):
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(id="job-123")

    monkeypatch.setattr(worker, "queue_db", _fake_queue_db)
    prediction_id = prediction_with_cached_global
    response = client.post(
        f"/v1/xai/predictions/{prediction_id}/explanations/run",
        json={"xaiMethod": "shap_auto", "outputStatistic": "median", "topK": 5},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"id": "job-123"}
    assert captured["kwargs"]["__job_name__"] == "cached prediction shap_auto"
    assert captured["kwargs"]["__prediction_id__"] == prediction_id
    assert captured["kwargs"]["__xai_method__"] == "shap_auto"
