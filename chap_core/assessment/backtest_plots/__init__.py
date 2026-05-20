"""
Backtest visualization plugin system.

This module provides a clean interface for creating custom backtest visualizations.
Each visualization is a single class that receives flat pandas DataFrames and returns
an Altair chart.

Example usage:
    from chap_core.assessment.backtest_plots import backtest_plot, BacktestPlotBase
    import pandas as pd
    import altair as alt

    @backtest_plot(
        plot_id="my_custom_plot",
        name="My Custom Plot",
        description="A custom visualization showing forecast accuracy."
    )
    class MyCustomPlot(BacktestPlotBase):
        def plot(
            self,
            observations: pd.DataFrame,
            forecasts: pd.DataFrame,
            historical_observations: pd.DataFrame | None = None
        ) -> alt.Chart:
            # observations has columns: location, time_period, disease_cases
            # forecasts has columns: location, time_period, horizon_distance, sample, forecast
            # historical_observations (optional): location, time_period, disease_cases
            ...
            return chart

Data schemas:
    observations (pd.DataFrame):
        - location: str - Location identifier
        - time_period: str - Time period (e.g., "2024-01" or "202401")
        - disease_cases: float - Observed disease cases

    forecasts (pd.DataFrame):
        - location: str - Location identifier
        - time_period: str - Time period being forecasted
        - horizon_distance: int - How many periods ahead this forecast is
        - sample: int - Sample index (for probabilistic forecasts)
        - forecast: float - Forecasted value

    historical_observations (pd.DataFrame, optional):
        - location: str - Location identifier
        - time_period: str - Time period
        - disease_cases: float - Historical observed disease cases
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import altair as alt

if TYPE_CHECKING:
    import pandas as pd

    from chap_core.database.tables import Backtest

# Type alias for Altair chart types that plots can return
ChartType = alt.Chart | alt.VConcatChart | alt.FacetChart | alt.LayerChart | alt.HConcatChart

# Global registry for backtest plots
_backtest_plots_registry: dict[str, type[BacktestPlotBase]] = {}


class BacktestPlotBase(ABC):
    """
    Base class for backtest visualizations.

    Subclasses must implement the `plot` method which receives flat DataFrames
    and returns an Altair chart.

    Attributes:
        id: Unique identifier for the plot (set by decorator)
        name: Display name for the plot (set by decorator)
        description: Description of what the plot shows (set by decorator)
        needs_historical: Whether this plot needs historical observations (set by decorator)
    """

    id: str = ""
    name: str = ""
    description: str = ""
    needs_historical: bool = False

    facet_dimensions: list[str] = []

    @abstractmethod
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        """
        Generate the visualization from flat DataFrames.

        Parameters
        ----------
        observations : pd.DataFrame
            Observed values with columns: location, time_period, disease_cases
        forecasts : pd.DataFrame
            Forecast samples with columns: location, time_period, horizon_distance,
            sample, forecast
        historical_observations : pd.DataFrame, optional
            Historical observations before split periods, with columns:
            location, time_period, disease_cases. Only provided if needs_historical=True.

        Returns
        -------
        ChartType
            Altair chart specification (Chart, VConcatChart, FacetChart, etc.)
        """

    def facet_coords(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None
    ) -> dict[str, list[Any]]:
        """Returns all unique coordinates available for faceting based on data fields."""
        coords: dict[str, list[Any]] = {}
        for dim in self.facet_dimensions:
            values = set()
            if dim == "split_period" and "split_period" not in forecasts.columns:
                if "horizon_distance" in forecasts.columns and "time_period" in forecasts.columns:
                    from chap_core.plotting.backtest_plot import clean_time
                    from chap_core.time_period import TimePeriod
                    for _, row in forecasts[["time_period", "horizon_distance"]].drop_duplicates().iterrows():
                        try:
                            tp = TimePeriod.parse(str(row["time_period"]))
                            sp = tp - (int(row["horizon_distance"]) * tp.time_delta)
                            values.add(clean_time(sp.to_string()))
                        except Exception:
                            continue
            else:
                for df in [observations, forecasts, historical_observations]:
                    if df is not None and dim in df.columns:
                        values.update(df[dim].dropna().unique())
            coords[dim] = sorted(values)
        return coords

    def get_subplot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        coords: dict[str, Any],
        historical_observations: pd.DataFrame | None = None,
    ) -> alt.Chart:
        """Slices data down to target coordinates for isolated rendering frames."""
        obs_df, fc_df = observations.copy(), forecasts.copy()
        hist_df = historical_observations.copy() if historical_observations is not None else None

        # Dynamically filter any dataframe columns that match the target coordinates
        for key, value in coords.items():
            if key in obs_df.columns:
                obs_df = obs_df[obs_df[key] == value]
            if key in fc_df.columns:
                fc_df = fc_df[fc_df[key] == value]
            if hist_df is not None and key in hist_df.columns:
                hist_df = hist_df[hist_df[key] == value]

        return self.get_full_plot(obs_df, fc_df, hist_df)

    def get_subplots(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        coords: dict[str, list[Any]],
        historical_observations: pd.DataFrame | None = None,
    ) -> list[tuple[Any, alt.Chart]]:
        """Generates a list of subplots paired with their respective coordinate value keys."""


        keys = list(coords.keys())
        value_lists = [coords[k] for k in keys]

        results = []
        # Generate the Cartesian product of all coordinate dimension values
        for combinations in itertools.product(*value_lists):
            single_coord = dict(zip(keys, combinations,strict=True))
            chart = self.get_subplot(observations, forecasts, single_coord, historical_observations)

            # If there's only one facet dimension, pass the raw single value as the key.
            # Otherwise, pass the combination tuple so it's hashable for set comparisons.
            key = combinations[0] if len(combinations) == 1 else combinations

            results.append((key, chart))

        return results

    def get_full_plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        """Applies programmatic grid faceting to visual configurations on local environments."""
        chart = self.plot(observations, forecasts, historical_observations)
        if self.facet_dimensions:
            facet_kwargs = {}
            if "location" in self.facet_dimensions:
                facet_kwargs["row"] = "location:N"
            if "split_period" in self.facet_dimensions:
                facet_kwargs["column"] = "split_period:O"
            return chart.facet(**facet_kwargs).resolve_scale(y="independent")
        return chart


def backtest_plot(
    plot_id: str,
    name: str,
    description: str = "",
    needs_historical: bool = False,
):
    """
    Decorator to register a backtest plot class.

    Parameters
    ----------
    plot_id : str
        Unique identifier for the plot (used in URLs/APIs)
    name : str
        Human-readable display name
    description : str, optional
        Description of what the plot shows
    needs_historical : bool, optional
        Whether this plot needs historical observations for context.
        Default is False.
    """

    def decorator(cls: type[BacktestPlotBase]) -> type[BacktestPlotBase]:
        if not issubclass(cls, BacktestPlotBase):
            raise TypeError(f"{cls.__name__} must inherit from BacktestPlotBase")

        cls.id = plot_id
        cls.name = name
        cls.description = description
        cls.needs_historical = needs_historical

        _backtest_plots_registry[plot_id] = cls
        return cls

    return decorator


def get_backtest_plots_registry() -> dict[str, type[BacktestPlotBase]]:
    """Get the registry of all registered backtest plots."""
    return _backtest_plots_registry.copy()


def get_backtest_plot(plot_id: str) -> type[BacktestPlotBase] | None:
    """Get a specific backtest plot class by ID."""
    return _backtest_plots_registry.get(plot_id)


def list_backtest_plots() -> list[dict]:
    """List all registered backtest plots with their metadata."""
    return [
        {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
            "needs_historical": cls.needs_historical,
        }
        for cls in _backtest_plots_registry.values()
    ]


def create_plot_from_backtest(plot_id: str, backtest: Backtest) -> ChartType:
    """Create a plot from a Backtest object."""
    from chap_core.assessment.evaluation import Evaluation

    plot_cls = get_backtest_plot(plot_id)
    if plot_cls is None:
        available = ", ".join(_backtest_plots_registry.keys())
        raise ValueError(f"Unknown plot type: {plot_id}. Available: {available}")

    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()

    observations_df: pd.DataFrame = flat_data.observations  # type: ignore[assignment]
    forecasts_df: pd.DataFrame = flat_data.forecasts  # type: ignore[assignment]

    historical_df: pd.DataFrame | None = None
    if plot_cls.needs_historical and flat_data.historical_observations is not None:
        historical_df = flat_data.historical_observations  # type: ignore[assignment]

    plotter = plot_cls()
    return plotter.get_full_plot(observations_df, forecasts_df, historical_df)


def create_plot_from_evaluation(plot_id: str, evaluation) -> ChartType:
    """Create a plot from an Evaluation object."""
    plot_cls = get_backtest_plot(plot_id)
    if plot_cls is None:
        available = ", ".join(_backtest_plots_registry.keys())
        raise ValueError(f"Unknown plot type: {plot_id}. Available: {available}")

    flat_data = evaluation.to_flat()

    observations_df: pd.DataFrame = flat_data.observations  # type: ignore[assignment]
    forecasts_df: pd.DataFrame = flat_data.forecasts  # type: ignore[assignment]

    historical_df: pd.DataFrame | None = None
    if plot_cls.needs_historical and flat_data.historical_observations is not None:
        historical_df = flat_data.historical_observations  # type: ignore[assignment]

    plotter = plot_cls()
    return plotter.get_full_plot(observations_df, forecasts_df, historical_df)


def _discover_plots():
    """Import all plot modules to trigger decorator registration."""
    from chap_core.assessment.backtest_plots import (
        evaluation_plot,
        horizon_location_grid,
        metrics_dashboard,
        predicted_vs_actual_linear_plot,
        predicted_vs_actual_plot,
        sample_bias_plot,
    )


_discover_plots()
