import pytest
from sqlmodel import Session
from starlette.testclient import TestClient

from chap_core.database.dataset_tables import DataSet
from chap_core.database.tables import BackTestRead
from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.dependencies import get_session

class DirectClient(TestClient):
    def get_json(self, *args, **kwargs):
        response = self.get(*args, **kwargs)
        assert response.status_code == 200, response.json()
        return response.json()

    def get_obj(self, *args, **kwargs):
        model= kwargs.pop('__model__')
        response = self.get_json(*args, **kwargs)
        return model.model_validate(response)



client = DirectClient(app)

@pytest.fixture
def override_session(p_seeded_engine):
    def get_test_session():
        with Session(p_seeded_engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    yield
    app.dependency_overrides.clear()

def test_dataset(seeded_session):
    dataset = seeded_session.query(DataSet)
    assert dataset[0].data_sources[0].covariate == 'mean_temperature'
    assert dataset.count() == 1


def test_get_evaluation_entries(override_session):
    params = {"backtestId": 1, "quantiles": [0.1, 0.5, 0.9]}
    evaluation_entries = client.get_json("/v1/analytics/evaluation-entry", params=params)
    assert len(evaluation_entries) > 3

def test_get_prediction_entries(override_session):
    params = {"predictionId": 1, "quantiles": [0.0, 0.5, 0.9]}
    prediction_entries = client.get_json("/v1/analytics/prediction-entry", params=params)
    assert len(prediction_entries) > 3

def test_get_backtest(override_session):
    backtest: BackTestRead = client.get_obj("/v1/crud/backtests/1/info", __model__ = BackTestRead)
    dataset = backtest.dataset
    assert dataset.data_sources[0].covariate == 'mean_temperature'
    assert len(dataset.org_units) == 3, dataset.org_units
    assert dataset.first_period
    assert dataset.last_period
