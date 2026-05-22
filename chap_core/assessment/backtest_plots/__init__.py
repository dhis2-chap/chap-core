"""
Backtest visualization plugin system.

This module provides a clean interface for creating custom backtest visualizations.
Each visualization is a single class that receives flat pandas DataFrames and returns
an Altair chart.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import altair as alt

if TYPE_CHECKING:
    import pandas as pd

    from chap_core.database.tables import Backtest

# Type alias for Altair chart types that plots can return
ChartType = alt.Chart | alt.VConcatChart | alt.FacetChart | alt.LayerChart | alt.HConcatChart

# Global registry for backtest plots
_backtest_plots_registry: dict[str, type[BacktestPlotBase]] = {}


@dataclass
class FacetDimension:
    """Structured metadata for facet dimensions, matching your visual configuration pipeline."""
    field_name: str
    display_name: str

    @property
    def clean_name(self) -> str:
        """Returns the raw column name without Altair type suffixes for internal processing."""
        return self.field_name.split(":", 1)[0]


class BacktestPlotBase(ABC):
    """
    Base class for backtest visualizations.

    Subclasses must implement the `plot` method which receives flat DataFrames
    and returns an Altair chart.
    """

    id: str = ""
    name: str = ""
    description: str = ""
    needs_historical: bool = False
    facet_dimensions: list[FacetDimension] = []

    @abstractmethod
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        """Generate the visualization from flat DataFrames."""

    def get_full_plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        """
        Pass-through fallback for non-faceted plots.
        Ensures incremental migration compatibility across the registry.
        """
        return self.plot(observations, forecasts, historical_observations)


class FacetedBacktestPlot(BacktestPlotBase):
    """
    Subclass managing generic programmatic grid faceting and subplots.
    Separates concerns into a definitive data preprocessing and visual layout pipeline.
    """

    facet_dimensions: list[FacetDimension] = []

    # Overridable scale resolutions configuration flags for custom plot types
    resolve_scale_x: str = "shared"
    resolve_scale_y: str = "independent"

    @abstractmethod
    def _preprocess(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Combine inputs into a single dataframe carrying every column the plot consumes."""

    @abstractmethod
    def _plot(self, df: pd.DataFrame) -> ChartType:
        """Render the unfaceted base chart from the preprocessed dataframe."""

    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        return self._plot(self._preprocess(observations, forecasts, historical_observations))

    def facet_coords(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> dict[str, list[Any]]:
        """Returns unique coordinates available for faceting from preprocessed fields."""
        df = self._preprocess(observations, forecasts, historical_observations)
        clean_dims = [dim.clean_name for dim in self.facet_dimensions]

        return {col: sorted(df[col].dropna().unique()) for col in clean_dims if col in df.columns}

    def get_subplot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        coords: dict[str, Any],
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        """Filters preprocessed data frames directly for isolated frame rendering."""
        df = self._preprocess(observations, forecasts, historical_observations)
        for col, value in coords.items():
            if col in df.columns:
                df = df[df[col] == value]
        return self._plot(df)

    def get_subplots(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        coords: dict[str, list[Any]],
        historical_observations: pd.DataFrame | None = None,
    ) -> list[tuple[Any, ChartType]]:
        """Generates subplots mapped directly against their Cartesian matrix values."""
        df_preprocessed = self._preprocess(observations, forecasts, historical_observations)

        keys = list(coords.keys())
        value_lists = [coords[k] for k in keys]
        results = []

        for combination in itertools.product(*value_lists):
            single_coords = dict(zip(keys, combination, strict=True))

            df_filtered = df_preprocessed
            for col, value in single_coords.items():
                if col in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered[col] == value]

            chart = self._plot(df_filtered)
            key = combination[0] if len(combination) == 1 else combination
            results.append((key, chart))
        return results

    def get_full_plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        """Applies layout grid faceting strategies safely across custom dimension arrays."""
        chart = self.plot(observations, forecasts, historical_observations)
        if not self.facet_dimensions:
            return chart

        # Safely extract dimension configurations explicitly
        col_dim = self.facet_dimensions[0] if len(self.facet_dimensions) > 0 else None
        row_dim = self.facet_dimensions[1] if len(self.facet_dimensions) > 1 else None

        # Handle native Altair types cleanly without local variable scoping issues
        if col_dim and row_dim:
            faceted_chart = chart.facet(
                column=alt.Column(col_dim.field_name, header=alt.Header(title=col_dim.display_name)),
                row=alt.Row(row_dim.field_name, header=alt.Header(title=row_dim.display_name))
            )
        elif col_dim:
            faceted_chart = chart.facet(
                column=alt.Column(col_dim.field_name, header=alt.Header(title=col_dim.display_name))
            )
        else:
            faceted_chart = chart

        resolve_chart: ChartType = faceted_chart.resolve_scale(
            x=self.resolve_scale_x,
            y=self.resolve_scale_y
        )
        return resolve_chart


def backtest_plot(
    plot_id: str,
    name: str,
    description: str = "",
    needs_historical: bool = False,
):
    """Decorator to register a backtest plot class."""

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
    return _backtest_plots_registry.copy()


def get_backtest_plot(plot_id: str) -> type[BacktestPlotBase] | None:
    return _backtest_plots_registry.get(plot_id)


def list_backtest_plots() -> list[dict]:
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
    from chap_core.assessment.backtest_plots import (
        evaluation_plot,
        horizon_location_grid,
        metrics_dashboard,
        predicted_vs_actual_linear_plot,
        predicted_vs_actual_plot,
        sample_bias_plot,
    )


_discover_plots()
