import json
from pathlib import Path

import pandas as pd
import pytest

from chap_core.assessment.metrics.rmse import DetailedRMSE
from chap_core.database.tables import BackTestMetric
from chap_core.plotting.evaluation_plot import MetricByHorizonV2, MetricMapV2, make_plot_from_backtest_object


@pytest.fixture
def rwanda_geojson():
    path = Path("/Users/knutdr/Data/ch_data/rainfall_pop_temp_cases.json")
    path = Path("rainfall_pop_temp_cases.json")
    if not path.exists():
        pytest.skip("Local data not found")
    d = json.loads(path.read_text())
    return d["geojson"]


@pytest.fixture
def rwanda_orgunits(rwanda_geojson) -> list[str]:
    return [feature["id"] for feature in rwanda_geojson["features"]]


@pytest.fixture
def rwanda_metrics(rwanda_orgunits) -> list[BackTestMetric]:
    time_periods = ["2022-02", "2022-03"]
    rows = [
        {"location": ou, "time_period": tp, "horizon_distance": 1, "metric": float((i * o + o) % 5)}
        for i, tp in enumerate(time_periods)
        for o, ou in enumerate(rwanda_orgunits)
    ]
    return pd.DataFrame(rows)


def test_plot_from_df(rwanda_geojson, rwanda_metrics):
    MetricMapV2(rwanda_metrics, rwanda_geojson).plot()
    # feature_props = rwanda_geojson['features'][0]['properties']

    # print(feature_props)
    # assert False


def test_evaluation_plot_from_backtest_object(backtest):
    plot = make_plot_from_backtest_object(backtest, MetricByHorizonV2, DetailedRMSE())
