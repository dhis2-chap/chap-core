import json

import altair
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
        model = kwargs.pop("__model__")
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


def test_dataset(seeded_session: Session):
    dataset = seeded_session.query(DataSet)
    assert dataset[0].data_sources[0].covariate == "mean_temperature"
    assert dataset[0].period_type == "month"
    assert dataset.count() == 2
    assert not dataset[1].data_sources
    assert len(dataset[1].observations) > 0


def test_get_evaluation_entries(override_session):
    params = {"backtestId": 1, "quantiles": [0.1, 0.5, 0.9]}
    evaluation_entries = client.get_json("/v1/analytics/evaluation-entry", params=params)
    assert len(evaluation_entries) > 3


def test_get_prediction_entries(override_session):
    params = {"predictionId": 1, "quantiles": [0.0, 0.5, 0.9]}
    prediction_entries = client.get_json("/v1/analytics/prediction-entry", params=params)
    assert len(prediction_entries) > 3


def test_get_backtest(override_session):
    backtest: BackTestRead = client.get_obj("/v1/crud/backtests/1/info", __model__=BackTestRead)
    dataset = backtest.dataset
    assert dataset.data_sources[0].covariate == "mean_temperature"
    assert len(dataset.org_units) == 3, dataset.org_units
    assert dataset.first_period
    assert dataset.last_period


def test_data_plot(override_session, tmp_path):
    response = client.get("/v1/plots/dataset/standardized-feature/2")
    assert response.status_code == 200, response.json()
    vega_spec = response.json()
    html_template = wrap_vega_spec(vega_spec)
    # with open(tmp_path/"chap_core_chart.html", "w") as f:
    #    f.write(html_template)

def test_backtest_plot(override_session, tmp_path):
    response = client.get("/v1/plots/backtest/tmp/1")
    assert response.status_code == 200, response.json()
    vega_spec = response.json()
    html_template = wrap_vega_spec(vega_spec)
    # with open(tmp_path/"chap_core_chart.html", "w") as f:
    #    f.write(html_template)

def wrap_vega_spec(vega_spec) -> str:
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    </head>
    <body>
        <div id="vis"></div>
        <script type="text/javascript">
            vegaEmbed('#vis', {json.dumps(vega_spec)});
        </script>
    </body>
    </html>
    """
    return html_template
