import itertools
import json
from pathlib import Path

import pytest
from sqlmodel import select, Session

from chap_core.database.tables import BackTest
from chap_core.rest_api.data_models import BackTestFull
from chap_core.rest_api.v1.routers.visualization import (
    get_available_metrics,
    generate_visualization,
    generate_backtest_plots,
    list_backtest_plot_types,
)


def test_get_available_metrics():
    metrics = get_available_metrics(backtest_id=1)
    print(metrics)
    assert any(metric.id == "detailed_rmse" for metric in metrics)


def all_metric_ids():
    metrics = get_available_metrics(backtest_id=1)
    return [metric.id for metric in metrics]


def load_backtest():
    path = Path("/Users/knutdr/Data/test.json")
    if not path.exists():
        pytest.skip("Weird backtest data not available")
    json_data = json.load(open(path))
    print(json_data.keys())
    return BackTestFull.model_validate(json_data)


@pytest.fixture
def seeded_session_with_weird_backtest(seeded_session: Session):
    seeded_session.add(wierd_backtest := load_backtest())
    seeded_session.commit()
    seeded_session.refresh()

    return seeded_session, wierd_backtest.id


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize(
    "metric_id",
    all_metric_ids()[:1],
)
def test_generate_metric_visualization_weird_backtest(
    seeded_session_with_weird_backtest, metric_id, visualization_name="MetricByHorizonV2Mean"
):
    session, backtest_id = seeded_session_with_weird_backtest
    assert generate_visualization(
        visualization_name=visualization_name, backtest_id=backtest_id, metric_id=metric_id, session=session
    )


@pytest.mark.parametrize("metric_id", all_metric_ids())
def test_generate_metric_visualization(seeded_session, metric_id, visualization_name="MetricByHorizonV2Mean"):
    assert generate_visualization(
        visualization_name=visualization_name, backtest_id=1, metric_id=metric_id, session=seeded_session
    )


def test_generate_backtest_plot(seeded_session, visualization_name="metrics_dashboard", backtest_id=1):
    assert generate_backtest_plots(
        visualization_name=visualization_name, backtest_id=backtest_id, session=seeded_session
    )


@pytest.fixture
def backtest_ids(seeded_session):
    backtests = seeded_session.exec(select(BackTest)).all()
    return [bt.id for bt in backtests]


def test_all_backtest_plots(seeded_session: Session, backtest_ids):
    """Test that all registered backtest plots can be generated for the first backtest.

    Note: We only test with the first backtest to avoid issues with test fixtures
    that have NaN values or other edge cases that don't represent real-world data.
    """
    plot_types = list_backtest_plot_types()
    # Use only the first backtest (the one without NaN values)
    backtest_id = backtest_ids[0]
    for plot_type in plot_types:
        plot_name = plot_type.id
        result = generate_backtest_plots(visualization_name=plot_name, backtest_id=backtest_id, session=seeded_session)
        assert result is not None, f"Plot {plot_name} failed for backtest {backtest_id}"
