from __future__ import annotations

import itertools

import altair as alt
import pytest

from chap_core.assessment.backtest_plots import FacetedBacktestPlot, get_backtest_plots_registry


@pytest.mark.parametrize("plot_name,plot_cls", list(get_backtest_plots_registry().items()))
def test_faceting_behavior_across_registry(plot_name, plot_cls, flat_observations, flat_forecasts):
    """Smoke test exercising the full faceting pipeline for every registered plot.

    Ensures that faceted plots expose the declared facet coordinates, can produce
    isolated subplots, a full Cartesian set of subplots, and a faceted full plot.
    """
    if not issubclass(plot_cls, FacetedBacktestPlot):
        pytest.skip(f"{plot_name} does not participate in faceting")

    plotter = plot_cls()

    # facet_coords must return a mapping of declared (clean) field names -> list of values
    coords = plotter.facet_coords(flat_observations, flat_forecasts)
    assert isinstance(coords, dict)

    # pick one value per dimension for get_subplot
    values_lists = list(coords.values())
    if not values_lists:
        pytest.skip("no facet dimensions exposed")

    selection = {k: v[0] for k, v in coords.items()}

    # get_subplot must return a TopLevel chart (not a FacetChart)
    chart = plotter.get_subplot(flat_observations, flat_forecasts, selection)
    assert isinstance(chart, alt.TopLevelMixin)
    assert not isinstance(chart, alt.FacetChart)

    # get_subplots must return a chart for each Cartesian product of coords
    subplots = plotter.get_subplots(flat_observations, flat_forecasts, coords)
    expected_count = 1
    for values in coords.values():
        expected_count *= len(values)
    assert isinstance(subplots, list)
    assert len(subplots) == expected_count
    for key, ch in subplots:
        assert isinstance(ch, alt.TopLevelMixin)

    # get_full_plot should yield a (possibly) faceted top-level chart
    full = plotter.get_full_plot(flat_observations, flat_forecasts)
    assert isinstance(full, alt.TopLevelMixin)
