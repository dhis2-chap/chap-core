import pytest
from sqlmodel import Session
from starlette.testclient import TestClient

from chap_core.database.dataset_tables import DataSet
from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.dependencies import get_session

class DirectClient(TestClient):
    def get_json(self, *args, **kwargs):
        response = self.get(*args, **kwargs)
        assert response.status_code == 200, response.json()
        return response.json()

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
    prediction_entries = client.get_json("/v1/analytics/prediction-entry/", params=params)
    assert len(prediction_entries) > 3
