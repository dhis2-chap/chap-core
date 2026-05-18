"""Smoke tests for the faceting API described in CLIM-548.

Custom backtest visualizations should expose facetable dimensions and let the
front-end request individual subplots instead of always rendering a single
vega spec that does its own faceting.

The expected public surface on a backtest plot class:

- ``facet_dimensions: list[str]`` -- class attribute declaring which
  dimensions (e.g. ``location``, ``horizon_distance``) can be faceted.
- ``facet_coords() -> dict[str, list]`` -- coordinate values available
  for each declared facet dimension, given the bound data.
- ``get_subplot(coords: dict[str, Any]) -> alt.Chart`` -- one subplot for
  the supplied coordinate selection. Only the data needed for that
  subplot should be included.
- ``get_subplots(coords: dict[str, list[Any]]) -> list[tuple[Any, alt.Chart]]``
  -- batched subplot rendering.
- ``get_full_plot() -> alt.Chart`` -- the full plot, used by the CLI.

The feature is not yet implemented, so each test is marked ``xfail`` with
``strict=True``. When the API lands, the tests will start passing and the
marker will need to be removed.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import pytest

from chap_core.assessment.backtest_plots import (
    BacktestPlotBase,
    get_backtest_plots_registry,
)
from chap_core.assessment.backtest_plots.evaluation_plot import EvaluationPlot


CLIM_548 = "CLIM-548: faceting API not implemented yet"


@pytest.fixture
def faceting_plotter(flat_observations, flat_forecasts) -> BacktestPlotBase:
    """Instantiate an EvaluationPlot bound to the standard flat fixtures.

    The exact data-binding mechanism (constructor kwargs, ``set_data`` call,
    factory function) is intentionally left to the implementer; this fixture
    will need to be updated to match the chosen design.
    """
    return EvaluationPlot(
        observations=pd.DataFrame(flat_observations),
        forecasts=pd.DataFrame(flat_forecasts),
    )


@pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_facet_dimensions_declared_on_all_registered_plots():
    """Every registered plot class must declare its facetable dimensions."""
    registry = get_backtest_plots_registry()
    assert registry, "expected at least one registered plot"
    for plot_id, plot_cls in registry.items():
        assert hasattr(plot_cls, "facet_dimensions"), (
            f"{plot_id} is missing facet_dimensions"
        )
        dims = plot_cls.facet_dimensions
        assert isinstance(dims, list)
        assert all(isinstance(d, str) for d in dims)


@pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_facet_coords_returns_values_per_dimension(faceting_plotter):
    coords = faceting_plotter.facet_coords()
    assert isinstance(coords, dict)
    for dim in faceting_plotter.facet_dimensions:
        assert dim in coords
        assert isinstance(coords[dim], list)
        assert len(coords[dim]) > 0


@pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_get_full_plot_returns_chart(faceting_plotter):
    chart = faceting_plotter.get_full_plot()
    assert isinstance(chart, alt.TopLevelMixin)


@pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_get_subplot_returns_chart_for_single_coord(faceting_plotter):
    coords = faceting_plotter.facet_coords()
    dim = next(iter(coords))
    selection = {dim: coords[dim][0]}
    chart = faceting_plotter.get_subplot(selection)
    assert isinstance(chart, alt.TopLevelMixin)


@pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_get_subplots_returns_one_chart_per_coordinate(faceting_plotter):
    coords = faceting_plotter.facet_coords()
    dim = next(iter(coords))
    values = coords[dim]
    subplots = faceting_plotter.get_subplots({dim: values})
    assert isinstance(subplots, list)
    assert len(subplots) == len(values)
    returned_keys = [key for key, _ in subplots]
    assert set(returned_keys) == set(values)
    for _, chart in subplots:
        assert isinstance(chart, alt.TopLevelMixin)
