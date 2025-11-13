import altair
import pytest

from chap_core.assessment.backtest_plots.sample_bias_plot import RatioOfSamplesAboveTruthBacktestPlot
from chap_core.database.tables import BackTest
from chap_core.plotting.backtest_plot import EvaluationBackTestPlot
from chap_core.assessment.backtest_plots.vega_lite_dash import combined_dashboard_from_backtest
from chap_core.assessment.backtest_plots.backtest_plot_1 import BackTestPlot1


@pytest.fixture(scope="module")
def default_transformer():
    altair.data_transformers.enable("default")
    yield


def test_backtest_plot(simulated_backtest: BackTest, default_transformer):
    plotter = EvaluationBackTestPlot.from_backtest(simulated_backtest)
    chart = plotter.plot()


def test_combined_dashboard_from_backtest(flat_observations, flat_forecasts, title="testplot"):
    plot = combined_dashboard_from_backtest(flat_observations, flat_forecasts, title)
    # plot.show()


def test_sample_bias_plot(simulated_backtest):
    plotter = RatioOfSamplesAboveTruthBacktestPlot.from_backtest(simulated_backtest)
    chart = plotter.plot()
    # chart.show()


def test_backtest_plot1(simulated_backtest):
    plotter = BackTestPlot1.from_backtest(simulated_backtest)
    chart = plotter.plot()
