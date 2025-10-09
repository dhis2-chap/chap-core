import altair

from chap_core.database.tables import BackTest
from chap_core.plotting.backtest_plot import BackTestPlot


def test_backtest_plot(simulated_backtest: BackTest):
    plotter = BackTestPlot.from_backtest(simulated_backtest)
    chart = plotter.plot()
