"""Smoke tests for the faceting API described in CLIM-548.

Backtest plots that participate in the faceting workflow live under a new
``FacetedBacktestPlot`` subclass of ``BacktestPlotBase``. The subclass forces
each plot to split work into two steps:

- ``_preprocess(observations, forecasts, historical_observations=None)
  -> pd.DataFrame`` -- combine the inputs into a single dataframe that
  includes every column the plot consumes, including derived dimensions
  like ``split_period``.
- ``_plot(df) -> alt.Chart`` -- render the unfaceted base chart from that
  dataframe.

The base class drives the rest. ``facet_dimensions`` is a list of altair
field shorthands such as ``["location:N", "split_period:O"]`` -- the
field name and the encoding type marker. With the preprocessed dataframe
in hand the base class can:

- compute ``facet_coords`` by reading unique values of each declared column;
- slice the preprocessed dataframe to a single coordinate selection for
  ``get_subplot`` / ``get_subplots``;
- apply the right altair ``.facet(...)`` call generically in
  ``get_full_plot`` based on the shorthands.

The base ``BacktestPlotBase`` itself stays minimal: it must not contain any
knowledge of specific facet dimension names.

Smoke tests run against more than one plot subclass to make sure the API is
not implicitly tied to ``EvaluationPlot``'s dimensions: a second plot with a
different facet dimension (``horizon_distance``) is included.

Tests that exercise structure introduced by the refactor (the new
subclass, the ``_preprocess`` / ``_plot`` split, the type-marker shorthand
in ``facet_dimensions``, the second plot opting in) are marked
``xfail(strict=True)`` and must lose the marker once the refactor lands.
Tests that exercise behavior already provided by PR 353 (``facet_coords``,
``get_subplot``, ``get_subplots``, ``get_full_plot``) run live -- they
double as regression checks that the refactor preserves observable
behavior.
"""

from __future__ import annotations

import re

import altair as alt
import pandas as pd
import pytest

from chap_core.assessment.backtest_plots import (
    BacktestPlotBase,
    get_backtest_plots_registry,
)
from chap_core.assessment.backtest_plots.evaluation_plot import EvaluationPlot
from chap_core.assessment.backtest_plots.predicted_vs_actual_plot import (
    PredictedVsActualPlot,
)


CLIM_548 = "CLIM-548: FacetedBacktestPlot refactor not implemented yet"

# Altair shorthand: ``field:T`` where T is one of N (nominal), O (ordinal),
# Q (quantitative), T (temporal).
FACET_DIM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:[NOQT]$")


@pytest.fixture
def observations_df(flat_observations) -> pd.DataFrame:
    return pd.DataFrame(flat_observations)


@pytest.fixture
def forecasts_df(flat_forecasts) -> pd.DataFrame:
    return pd.DataFrame(flat_forecasts)


# Plot classes covered by the parametrized smoke tests. Each entry contains
# a class and the facet field names (without type markers) that it must
# expose so subplots can be sliced from the preprocessed dataframe.
PLOT_CASES = [
    pytest.param(EvaluationPlot, ("location", "split_period"), id="evaluation_plot"),
    pytest.param(
        PredictedVsActualPlot, ("horizon_distance",), id="predicted_vs_actual"
    ),
]


@pytest.fixture
def faceted_base():
    """The new intermediate base class introduced by the refactor.

    Imported lazily so the xfail marker can catch the ImportError that
    occurs while the refactor is unimplemented.
    """
    from chap_core.assessment.backtest_plots import FacetedBacktestPlot

    return FacetedBacktestPlot


# --- Base class shape ----------------------------------------------------


@pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_faceted_base_class_is_a_subclass_of_backtest_plot_base(faceted_base):
    assert issubclass(faceted_base, BacktestPlotBase)
    assert faceted_base is not BacktestPlotBase


@pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_base_class_does_not_know_specific_facet_dimensions():
    """``BacktestPlotBase`` itself must not reference any specific facet
    dimension name (``location``, ``split_period``, ``horizon_distance``)
    in source -- generic faceting lives on the FacetedBacktestPlot subclass.
    """
    import inspect

    from chap_core.assessment.backtest_plots import BacktestPlotBase as _Base

    source = inspect.getsource(_Base)
    for forbidden in ("location", "split_period", "horizon_distance"):
        assert forbidden not in source, (
            f"BacktestPlotBase references {forbidden!r}; faceting knowledge "
            "should live on FacetedBacktestPlot instead."
        )


# --- Per-plot shape ------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=CLIM_548)
@pytest.mark.parametrize("plot_cls,expected_fields", PLOT_CASES)
def test_plot_inherits_from_faceted_base(plot_cls, expected_fields, faceted_base):
    assert issubclass(plot_cls, faceted_base)


@pytest.mark.xfail(strict=True, reason=CLIM_548)
@pytest.mark.parametrize("plot_cls,expected_fields", PLOT_CASES)
def test_facet_dimensions_use_type_marker_shorthand(plot_cls, expected_fields):
    dims = plot_cls.facet_dimensions
    assert isinstance(dims, list)
    assert dims, f"{plot_cls.__name__} declares no facet dimensions"
    for entry in dims:
        assert isinstance(entry, str)
        assert FACET_DIM_RE.match(entry), (
            f"{plot_cls.__name__}.facet_dimensions entry {entry!r} must be "
            "an altair shorthand like 'location:N' or 'horizon_distance:O'"
        )
    declared_fields = tuple(entry.split(":", 1)[0] for entry in dims)
    assert set(declared_fields) == set(expected_fields), (
        f"{plot_cls.__name__} declared {declared_fields}, expected {expected_fields}"
    )


# --- Preprocessing + plotting split --------------------------------------


@pytest.mark.xfail(strict=True, reason=CLIM_548)
@pytest.mark.parametrize("plot_cls,expected_fields", PLOT_CASES)
def test_preprocess_returns_dataframe_with_all_facet_columns(
    plot_cls, expected_fields, observations_df, forecasts_df
):
    """``_preprocess`` must produce a dataframe carrying every facet column,
    including derived ones (e.g. ``split_period``). This is what lets the
    base class compute coords and slice subplots without knowing how each
    plot derives its columns."""
    plotter = plot_cls()
    df = plotter._preprocess(observations_df, forecasts_df)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for field in expected_fields:
        assert field in df.columns, (
            f"{plot_cls.__name__}._preprocess must expose {field!r} so the "
            "base class can read coordinates from the preprocessed dataframe"
        )


@pytest.mark.xfail(strict=True, reason=CLIM_548)
@pytest.mark.parametrize("plot_cls,expected_fields", PLOT_CASES)
def test_plot_renders_unfaceted_chart_from_preprocessed_dataframe(
    plot_cls, expected_fields, observations_df, forecasts_df
):
    plotter = plot_cls()
    df = plotter._preprocess(observations_df, forecasts_df)
    chart = plotter._plot(df)
    assert isinstance(chart, alt.TopLevelMixin)
    # The split between _plot and get_full_plot means _plot itself must not
    # produce a faceted chart -- the base class applies .facet(...).
    assert not isinstance(chart, alt.FacetChart)


# --- Faceting API --------------------------------------------------------


@pytest.mark.parametrize("plot_cls,expected_fields", PLOT_CASES)
def test_facet_coords_returns_values_per_dimension(
    plot_cls, expected_fields, observations_df, forecasts_df
):
    """Behavioral regression: faceting must keep working across the refactor."""
    if not plot_cls.facet_dimensions:
        pytest.xfail(CLIM_548)
    plotter = plot_cls()
    coords = plotter.facet_coords(observations_df, forecasts_df)
    assert isinstance(coords, dict)
    assert set(coords.keys()) == set(expected_fields)
    for field, values in coords.items():
        assert isinstance(values, list)
        assert len(values) > 0
        assert len(set(values)) == len(values), f"duplicate coords for {field}"


@pytest.mark.xfail(strict=True, reason=CLIM_548)
@pytest.mark.parametrize("plot_cls,expected_fields", PLOT_CASES)
def test_get_subplot_returns_chart_for_single_coord(
    plot_cls, expected_fields, observations_df, forecasts_df
):
    """A subplot is a single cell, not a facet wrapper -- today's
    ``get_subplot`` calls ``get_full_plot`` and so re-applies ``.facet(...)``,
    producing a degenerate 1x1 ``FacetChart``. After the refactor a subplot
    must come straight out of ``_plot`` with no facet wrapping."""
    plotter = plot_cls()
    coords = plotter.facet_coords(observations_df, forecasts_df)
    selection = {field: coords[field][0] for field in expected_fields}
    chart = plotter.get_subplot(observations_df, forecasts_df, selection)
    assert isinstance(chart, alt.TopLevelMixin)
    assert not isinstance(chart, alt.FacetChart)


@pytest.mark.parametrize("plot_cls,expected_fields", PLOT_CASES)
def test_get_subplots_returns_one_chart_per_coordinate_combination(
    plot_cls, expected_fields, observations_df, forecasts_df
):
    if not plot_cls.facet_dimensions:
        pytest.xfail(CLIM_548)
    plotter = plot_cls()
    coords = plotter.facet_coords(observations_df, forecasts_df)
    expected_count = 1
    for field in expected_fields:
        expected_count *= len(coords[field])
    subplots = plotter.get_subplots(observations_df, forecasts_df, coords)
    assert isinstance(subplots, list)
    assert len(subplots) == expected_count
    for _, chart in subplots:
        assert isinstance(chart, alt.TopLevelMixin)


@pytest.mark.parametrize("plot_cls,expected_fields", PLOT_CASES)
def test_get_full_plot_returns_faceted_chart(
    plot_cls, expected_fields, observations_df, forecasts_df
):
    plotter = plot_cls()
    chart = plotter.get_full_plot(observations_df, forecasts_df)
    assert isinstance(chart, alt.TopLevelMixin)



# --- Registry-level invariant -------------------------------------------


@pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_every_faceted_plot_in_registry_passes_shape_checks(faceted_base):
    """In time every backtest plot should be a FacetedBacktestPlot. For now
    this test just asserts that at least one plot has been migrated and that
    every migrated plot satisfies the shorthand shape rule."""
    registry = get_backtest_plots_registry()
    faceted = [cls for cls in registry.values() if issubclass(cls, faceted_base)]
    assert faceted, "expected at least one FacetedBacktestPlot in the registry"
    for cls in faceted:
        dims = cls.facet_dimensions
        assert isinstance(dims, list) and dims
        for entry in dims:
            assert FACET_DIM_RE.match(entry), (
                f"{cls.__name__}.facet_dimensions entry {entry!r} is not "
                "altair shorthand"
            )
