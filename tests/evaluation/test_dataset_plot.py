import pytest

from chap_core.plotting.dataset_plot import (
    DatasetPlot,
    DiseaseCasesMap,
    StandardizedFeaturePlot,
    create_plot_from_dataset,
    get_dataset_plots_registry,
    list_dataset_plots,
)
from chap_core.plotting.season_plot import SeasonCorrelationBarPlot, SeasonalityComparisonPlot


@pytest.mark.parametrize(
    "plt_cls", [DiseaseCasesMap, StandardizedFeaturePlot, SeasonCorrelationBarPlot, SeasonalityComparisonPlot]
)
def test_standardized_feautre_plot(simulated_dataset, plt_cls):
    plotter = plt_cls.from_dataset_model(simulated_dataset)
    chart = plotter.plot()


def test_dataset_plot_registry():
    registry = get_dataset_plots_registry()
    expected_ids = {
        "disease-cases-map",
        "standardized-feature-plot",
        "seasonal-correlation-plot",
        "seasonality-comparison",
    }
    assert expected_ids.issubset(registry.keys())
    for cls in registry.values():
        assert issubclass(cls, DatasetPlot)


def test_list_dataset_plots():
    plots = list_dataset_plots()
    assert len(plots) >= 4
    for plot in plots:
        assert "id" in plot
        assert "name" in plot
        assert "description" in plot


@pytest.mark.parametrize("plot_id", list(get_dataset_plots_registry().keys()))
def test_create_plot_from_dataset(simulated_dataset, plot_id):
    spec = create_plot_from_dataset(plot_id, simulated_dataset)
    assert isinstance(spec, dict)
