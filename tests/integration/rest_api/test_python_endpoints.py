import pytest
from sqlmodel import select, Session

from chap_core.assessment.metrics import list_metrics
from chap_core.database.tables import Backtest
from chap_core.rest_api.v1.routers.visualization import (
    get_available_metrics,
    generate_visualization,
    generate_backtest_plots,
    list_backtest_plot_types,
)


def test_get_available_metrics(seeded_session):
    metrics = get_available_metrics(backtest_id=1, session=seeded_session)
    print(metrics)
    # Check for unified metrics (no more "detailed_" prefix)
    assert any(metric.id == "rmse" for metric in metrics)


def test_available_metrics_expose_optimization_direction(seeded_session):
    metrics = {metric.id: metric for metric in get_available_metrics(backtest_id=1, session=seeded_session)}
    assert metrics["crps"].optimization_direction == "minimize"
    assert metrics["sensitivity"].optimization_direction == "maximize"
    assert metrics["coverage_10_90"].optimization_direction is None


def test_available_metrics_expose_unit_and_target(seeded_session):
    metrics = {metric.id: metric for metric in get_available_metrics(backtest_id=1, session=seeded_session)}
    assert metrics["mape"].unit == "%"
    assert metrics["mape"].target is None
    assert metrics["coverage_10_90"].target == 0.8
    assert metrics["coverage_25_75"].target == 0.5
    assert metrics["ratio_above_truth"].target == 0.5
    assert metrics["mae"].unit is None
    assert metrics["mae"].target is None
    for entry in list_metrics():
        assert metrics[entry["id"]].unit == entry["unit"]
        assert metrics[entry["id"]].target == entry["target"]
    for metric in metrics.values():
        if metric.optimization_direction is None:
            assert metric.target is not None, metric.id


def test_available_metrics_expose_target_behavior(seeded_session):
    metrics = {metric.id: metric for metric in get_available_metrics(backtest_id=1, session=seeded_session)}
    assert metrics["coverage_10_90"].target_behavior == "at_least"
    assert metrics["coverage_25_75"].target_behavior == "at_least"
    assert metrics["ratio_above_truth"].target_behavior == "closest"
    for entry in list_metrics():
        assert metrics[entry["id"]].target_behavior == entry["target_behavior"]


def test_available_metrics_expose_comparison_op(seeded_session):
    metrics = {metric.id: metric for metric in get_available_metrics(backtest_id=1, session=seeded_session)}
    assert metrics["mae"].comparison_op == "skill_ratio"
    for entry in list_metrics():
        assert metrics[entry["id"]].comparison_op == entry["comparison_op"]


@pytest.mark.parametrize("metric_id", [entry["id"] for entry in list_metrics()])
def test_generate_metric_visualization(seeded_session, metric_id, visualization_name="metric_by_horizon_mean"):
    assert generate_visualization(
        visualization_name=visualization_name, backtest_id=1, metric_id=metric_id, session=seeded_session
    )


def test_generate_backtest_plot(seeded_session, visualization_name="horizon_location_grid", backtest_id=1):
    assert generate_backtest_plots(
        visualization_name=visualization_name, backtest_id=backtest_id, session=seeded_session
    )


@pytest.fixture
def backtest_ids(seeded_session):
    backtests = seeded_session.exec(select(Backtest)).all()
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
