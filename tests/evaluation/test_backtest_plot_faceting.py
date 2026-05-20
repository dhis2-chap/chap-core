"""Smoke tests for the faceting API described in CLIM-548.

Custom backtest visualizations should expose facetable dimensions and let the
front-end request individual subplots instead of always rendering a single
vega spec that does its own faceting.

The expected public surface on a backtest plot class:

- ``facet_dimensions: list[str]`` -- class attribute declaring which
  dimensions (e.g. ``location``, ``split_period``) can be faceted.
- ``facet_coords(observations, forecasts, historical_observations=None)
  -> dict[str, list]`` -- coordinate values available for each declared
  facet dimension, derived from the supplied data.
- ``get_subplot(observations, forecasts, coords,
  historical_observations=None) -> alt.Chart`` -- one subplot for the
  supplied coordinate selection. Only the data needed for that subplot
  should be included.
- ``get_subplots(observations, forecasts, coords,
  historical_observations=None) -> list[tuple[Any, alt.Chart]]``
  -- batched subplot rendering.
- ``get_full_plot(observations, forecasts, historical_observations=None)
  -> alt.Chart`` -- the full plot, used by the CLI.

Data flows through the methods as explicit parameters; the plot class does
not hold the data as instance attributes.

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
def observations_df(flat_observations) -> pd.DataFrame:
    return pd.DataFrame(flat_observations)


@pytest.fixture
def forecasts_df(flat_forecasts) -> pd.DataFrame:
    return pd.DataFrame(flat_forecasts)


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_facet_dimensions_declared_on_all_registered_plots():
    """Every registered plot class must declare its facetable dimensions."""
    registry = get_backtest_plots_registry()
    assert registry, "expected at least one registered plot"
    for plot_id, plot_cls in registry.items():
        assert hasattr(plot_cls, "facet_dimensions"), f"{plot_id} is missing facet_dimensions"
        dims = plot_cls.facet_dimensions
        assert isinstance(dims, list)
        assert all(isinstance(d, str) for d in dims)


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_facet_coords_returns_values_per_dimension(observations_df, forecasts_df):
    plotter: BacktestPlotBase = EvaluationPlot()
    coords = plotter.facet_coords(observations_df, forecasts_df)
    assert isinstance(coords, dict)
    for dim in plotter.facet_dimensions:
        assert dim in coords
        assert isinstance(coords[dim], list)
        assert len(coords[dim]) > 0


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_get_full_plot_returns_chart(observations_df, forecasts_df):
    plotter: BacktestPlotBase = EvaluationPlot()
    chart = plotter.get_full_plot(observations_df, forecasts_df)
    assert isinstance(chart, alt.TopLevelMixin)


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_get_subplot_returns_chart_for_single_coord(observations_df, forecasts_df):
    plotter: BacktestPlotBase = EvaluationPlot()
    coords = plotter.facet_coords(observations_df, forecasts_df)
    dim = next(iter(coords))
    selection = {dim: coords[dim][0]}
    chart = plotter.get_subplot(observations_df, forecasts_df, selection)
    assert isinstance(chart, alt.TopLevelMixin)


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_get_subplots_returns_one_chart_per_coordinate(observations_df, forecasts_df):
    plotter: BacktestPlotBase = EvaluationPlot()
    coords = plotter.facet_coords(observations_df, forecasts_df)
    dim = next(iter(coords))
    values = coords[dim]
    subplots = plotter.get_subplots(observations_df, forecasts_df, {dim: values})
    assert isinstance(subplots, list)
    assert len(subplots) == len(values)
    returned_keys = [key for key, _ in subplots]
    assert set(returned_keys) == set(values)
    for _, chart in subplots:
        assert isinstance(chart, alt.TopLevelMixin)


# --- EvaluationPlot-specific assertions -----------------------------------


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_evaluation_plot_facet_dimensions():
    assert EvaluationPlot.facet_dimensions == ["location", "split_period"]


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_evaluation_plot_facet_coords_match_unique_df_values(observations_df, forecasts_df):
    coords = EvaluationPlot().facet_coords(observations_df, forecasts_df)

    expected_locations = set(forecasts_df["location"].unique()) | set(observations_df["location"].unique())
    assert set(coords["location"]) == expected_locations
    assert len(coords["location"]) == len(set(coords["location"]))

    split_periods = coords["split_period"]
    assert len(split_periods) > 0
    assert len(split_periods) == len(set(split_periods))
