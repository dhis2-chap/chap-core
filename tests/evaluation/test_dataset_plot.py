import pytest

from chap_core.plotting.dataset_plot import StandardizedFeaturePlot
from chap_core.plotting.season_plot import SeasonCorrelationBarPlot


@pytest.fixture()
def default_transformer():
    import altair as alt
    alt.data_transformers.enable("default")
    yield

@pytest.mark.parametrize('plt_cls', [StandardizedFeaturePlot, SeasonCorrelationBarPlot])
def test_standardized_feautre_plot(simulated_dataset, plt_cls, default_transformer):
    plotter = plt_cls.from_dataset_model(simulated_dataset)
    chart = plotter.plot()
