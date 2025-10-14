import altair
import pytest

from chap_core.database.tables import BackTest
from chap_core.plotting.backtest_plot import BackTestPlot


@pytest.fixture(scope="module")
def default_transformer():
    altair.data_transformers.enable("default")
    yield


def test_backtest_plot(simulated_backtest: BackTest, default_transformer):
    plotter = BackTestPlot.from_backtest(simulated_backtest)
    chart = plotter.plot()
