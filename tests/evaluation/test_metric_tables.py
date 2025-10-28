import pytest
import pandas as pd
from chap_core.assessment.flat_representations import FlatMetric
from chap_core.database.tables import BackTest, BackTestMetric
from chap_core.assessment.metric_table import create_metric_table
from chap_core.plotting.evaluation_plot import MetricByHorizonV2Mean


def test_create_metric_table(backtest_metrics: list[BackTestMetric]):
    table = create_metric_table(backtest_metrics)

    assert len(table["last_seen_period"]) > 0
    assert table["horizon"].tolist()


# @pytest.fixture
# def metric_table(backtest_metrics: list[BackTestMetric]):
#   return create_metric_table(backtest_metrics)


@pytest.fixture
def flat_metric_data():
    return FlatMetric(
        pd.DataFrame(
            {
                "location": ["OrgUnit1", "OrgUnit1", "OrgUnit2", "OrgUnit2"],
                "horizon_distance": [1, 2, 1, 2],
                "time_period": ["2022-01", "2022-01", "2022-01", "2022-01"],
                "metric": [5.1, 6.2, 7.0, 8.0],
            }
        )
    )


def test_metric_plot_v2(flat_metric_data: FlatMetric):
    plotter = MetricByHorizonV2Mean(metric_data=flat_metric_data)
    chart = plotter.plot()
    print(chart)
