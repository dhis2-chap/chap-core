import pytest

from chap_core.database.tables import BackTest, BackTestMetric
from chap_core.assessment.metric_table import create_metric_table

def test_create_metric_table(backtest_metrics: list[BackTestMetric]):
    table = create_metric_table(backtest_metrics)
    assert len(table['value']) > 0
    assert len(table['last_seen_period']) > 0
    assert table['horizon'].tolist()


@pytest.fixture
def metric_table(backtest_metrics: list[BackTestMetric]):
    return create_metric_table(backtest_metrics)

def test_metric_plot(metric_table):
    from chap_core.plotting.evaluation_plot import MetricByHorizon
    plotter = MetricByHorizon()
    chart = plotter.plot_from_df(metric_table)
    chart.show()