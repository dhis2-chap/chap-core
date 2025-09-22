from chap_core.assessment.metrics import DetailedRMSE
from chap_core.plotting.evaluation_plot import MetricByHorizon, MetricByHorizonV2, make_plot_from_backtest_object


def test_evaluation_plot_from_backtest_object(backtest):
    plot = make_plot_from_backtest_object(backtest, MetricByHorizonV2, DetailedRMSE())
    # plot.show()
