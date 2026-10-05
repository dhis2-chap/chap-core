"""Stored-dataset target selection, including the actual worker/evaluation path."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from chap_core.assessment.evaluation import Evaluation
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSetCreateInfo
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.database.tables import Backtest, BacktestSpecification
from chap_core.datatypes import Samples
from chap_core.rest_api.app import app
from chap_core.rest_api.data_models import BacktestCreate, MakeBacktestRequest, MakeBacktestsRequest
from chap_core.rest_api.db_worker_functions import run_backtest
from chap_core.rest_api.v1.routers import analytics
from chap_core.rest_api.v1.routers.dependencies import get_session
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@pytest.fixture
def target_setup(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        dataset = DataSet.from_pandas(
            pd.DataFrame(
                [
                    {
                        "location": location,
                        "time_period": f"2020-{month:02d}",
                        "disease_cases": 100.0,
                        "custom_cases": float(month) if location == "A" else np.nan,
                        "rainfall": 2.0,
                    }
                    for location in ["A", "B"]
                    for month in range(1, 13)
                ]
            )
        )
        dataset_id = DataSetManager(session).save_dataset(DataSetCreateInfo(name="targets"), dataset, polygons=None)
        models = [
            ConfiguredModelDB(name=target, model_template=ModelTemplateDB(name=target, target=target, version="1"))
            for target in ["disease_cases", "incidence"]
        ]
        session.add_all(models)
        session.commit()
        worker = Mock()
        worker.queue_db.return_value = SimpleNamespace(id="job-1")
        monkeypatch.setattr(analytics, "worker", worker)
        app.dependency_overrides[get_session] = lambda: session
        yield TestClient(app), SessionWrapper(session=session), dataset_id, models, worker
        app.dependency_overrides.pop(get_session)
    engine.dispose()


@pytest.mark.parametrize(
    "request_model,model_field", [(MakeBacktestRequest, "modelId"), (MakeBacktestsRequest, "modelIds")]
)
def test_target_request_alias_and_default(request_model, model_field):
    payload = {"name": "run", "datasetId": 1, model_field: [1] if model_field == "modelIds" else 1}
    assert request_model.model_validate(payload).target_column is None
    request = request_model.model_validate({**payload, "targetColumn": "custom_cases"})
    assert request.target_column == "custom_cases"
    assert request.model_dump(by_alias=True)["targetColumn"] == "custom_cases"


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("target", [None, "custom_cases", "missing", "time_period"])
def test_target_validation_and_queueing(target_setup, multiple, target):
    client, wrapper, dataset_id, models, worker = target_setup
    payload = {
        "name": "run",
        "datasetId": dataset_id,
        "nPeriods": 2,
        "nSplits": 2,
        "modelIds" if multiple else "modelId": [m.id for m in models] if multiple else models[0].id,
    }
    if target is not None:
        payload["targetColumn"] = target
    response = client.post(
        "/v1/analytics/create-backtests" if multiple else "/v1/analytics/create-backtest", json=payload
    )
    if target in ("missing", "time_period"):
        assert response.status_code == 422, response.text
        assert target in response.json()["detail"]
        worker.queue_db.assert_not_called()
        assert wrapper.session.exec(select(BacktestSpecification)).all() == []
        return
    assert response.status_code == 200, response.text
    assert worker.queue_db.call_count == (2 if multiple else 1)
    for call in worker.queue_db.call_args_list:
        assert call.kwargs["target_column"] == target
    if multiple:
        specification = client.get(f"/v1/crud/backtest-specifications/{response.json()['specificationId']}").json()
        assert specification["targetColumn"] == (target or "disease_cases")
        assert set(specification["orgUnits"]) == ({"A"} if target else {"A", "B"})


@pytest.mark.parametrize("target", [None, "disease_cases", "custom_cases"])
def test_worker_uses_selected_truth_and_preserves_dataset(target_setup, monkeypatch, target):
    client, wrapper, dataset_id, models, worker = target_setup
    captures = []

    class RecordingEstimator:
        def __init__(self, target_name):
            self.target_name = target_name

        def train(self, data):
            captures.append(data)
            assert self.target_name in data.field_names()
            if target == "custom_cases":
                assert "custom_cases" not in data.field_names()
            return self

        def predict(self, historic_data, future_data):
            assert self.target_name in historic_data.field_names()
            assert self.target_name not in future_data.field_names()
            return DataSet(
                {
                    location: Samples(data.time_period, np.ones((len(data.time_period), 10)))
                    for location, data in future_data.items()
                }
            )

    monkeypatch.setattr(
        SessionWrapper,
        "get_configured_model_with_code",
        lambda self, model_id, prediction_length: RecordingEstimator(
            wrapper.session.get(ConfiguredModelDB, model_id).model_template.target
        ),
    )
    # The second model deliberately declares a different target; default behaviour is exercised with the first.
    run_models = models if target is not None else models[:1]
    specification_ids = set()
    for model in run_models:
        backtest_id = run_backtest(
            BacktestCreate(name="run", dataset_id=dataset_id, model_id=model.id),
            n_periods=2,
            n_splits=2,
            target_column=target,
            future_weather_provider="climatology" if target == "custom_cases" else "observed",
            session=wrapper,
        )
        backtest = wrapper.session.get(Backtest, backtest_id)
        specification_ids.add(backtest.specification_id)
        assert backtest.specification.target_column == (target or "disease_cases")
        flat = Evaluation.from_backtest(backtest).to_flat()
        expected = set(range(1, 13)) if target == "custom_cases" else {100.0}
        assert set(flat.observations.disease_cases.dropna()) == expected
        assert backtest.aggregate_metrics
    assert len(specification_ids) == 1
    for data in captures:
        target_name = "incidence" if "incidence" in data.field_names() else "disease_cases"
        assert getattr(data["A"], target_name)[0] == (1.0 if target == "custom_cases" else 100.0)
    original = DataSetManager(wrapper.session).to_dataset(dataset_id)
    assert original["A"].disease_cases[0] == 100.0
    assert original["A"].custom_cases[0] == 1.0


def test_different_targets_have_different_specifications(target_setup):
    client, wrapper, dataset_id, models, worker = target_setup
    payload = {"name": "run", "datasetId": dataset_id, "modelIds": [models[0].id]}
    default = client.post("/v1/analytics/create-backtests", json=payload).json()["specificationId"]
    explicit = client.post("/v1/analytics/create-backtests", json={**payload, "targetColumn": "disease_cases"}).json()[
        "specificationId"
    ]
    custom = client.post("/v1/analytics/create-backtests", json={**payload, "targetColumn": "custom_cases"}).json()[
        "specificationId"
    ]
    assert default == explicit
    assert custom != default
    rows = client.get("/v1/crud/backtest-specifications", params={"targetColumn": "custom_cases"}).json()
    assert [row["id"] for row in rows] == [custom]
    assert rows[0]["targetColumn"] == "custom_cases"


def test_no_training_target_data_is_rejected_before_queueing(target_setup):
    client, wrapper, dataset_id, models, worker = target_setup
    response = client.post(
        "/v1/analytics/create-backtests",
        json={
            "name": "run",
            "datasetId": dataset_id,
            "modelIds": [models[0].id],
            "targetColumn": "custom_cases",
            "nSplits": 12,
        },
    )
    assert response.status_code == 422, response.text
    assert "No org unit has target data" in response.json()["detail"]
    worker.queue_db.assert_not_called()


def test_custom_target_without_disease_cases_column(target_setup, monkeypatch):
    from chap_core.database.dataset_tables import DataSet as StoredDataSet
    from chap_core.predictor.naive_estimator import NaiveEstimator

    client, wrapper, dataset_id, models, worker = target_setup
    stored = wrapper.session.get(StoredDataSet, dataset_id)
    stored.covariates = [name for name in stored.covariates if name != "disease_cases"]
    for observation in stored.observations:
        if observation.feature_name == "disease_cases":
            wrapper.session.delete(observation)
    wrapper.session.commit()
    monkeypatch.setattr(SessionWrapper, "get_configured_model_with_code", lambda *args, **kwargs: NaiveEstimator())
    backtest_id = run_backtest(
        BacktestCreate(name="custom only", dataset_id=dataset_id, model_id=models[0].id),
        n_periods=2,
        n_splits=2,
        target_column="custom_cases",
        session=wrapper,
    )
    backtest = wrapper.session.get(Backtest, backtest_id)
    assert backtest.aggregate_metrics
    assert set(Evaluation.from_backtest(backtest).to_flat().observations.disease_cases.dropna()) == set(range(1, 13))
