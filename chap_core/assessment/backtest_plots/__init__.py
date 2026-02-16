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
        id="my_custom_plot",
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

from abc import ABC, abstractmethod

import altair as alt
import pandas as pd

from chap_core.database.tables import BackTest

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
        pass


def backtest_plot(
    id: str,
    name: str,
    description: str = "",
    needs_historical: bool = False,
):
    """
    Decorator to register a backtest plot class.

    Parameters
    ----------
    id : str
        Unique identifier for the plot (used in URLs/APIs)
    name : str
        Human-readable display name
    description : str, optional
        Description of what the plot shows
    needs_historical : bool, optional
        Whether this plot needs historical observations for context.
        Default is False.

    Example
    -------
    @backtest_plot(
        id="sample_bias",
        name="Sample Bias Plot",
        description="Shows forecast bias relative to observations"
    )
    class SampleBiasPlot(BacktestPlotBase):
        def plot(self, observations, forecasts, historical_observations=None):
            ...
    """

    def decorator(cls: type[BacktestPlotBase]) -> type[BacktestPlotBase]:
        if not issubclass(cls, BacktestPlotBase):
            raise TypeError(f"{cls.__name__} must inherit from BacktestPlotBase")

        cls.id = id
        cls.name = name
        cls.description = description
        cls.needs_historical = needs_historical

        _backtest_plots_registry[id] = cls
        return cls

    return decorator


def get_backtest_plots_registry() -> dict[str, type[BacktestPlotBase]]:
    """
    Get the registry of all registered backtest plots.

    Returns
    -------
    Dict[str, Type[BacktestPlotBase]]
        Dictionary mapping plot IDs to plot classes
    """
    return _backtest_plots_registry.copy()


def get_backtest_plot(plot_id: str) -> type[BacktestPlotBase] | None:
    """
    Get a specific backtest plot class by ID.

    Parameters
    ----------
    plot_id : str
        The unique identifier of the plot

    Returns
    -------
    Optional[Type[BacktestPlotBase]]
        The plot class, or None if not found
    """
    return _backtest_plots_registry.get(plot_id)


def list_backtest_plots() -> list[dict]:
    """
    List all registered backtest plots with their metadata.

    Returns
    -------
    list[dict]
        List of dicts with id, name, description, and needs_historical for each plot
    """
    return [
        {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
            "needs_historical": cls.needs_historical,
        }
        for cls in _backtest_plots_registry.values()
    ]


def create_plot_from_backtest(plot_id: str, backtest: BackTest) -> ChartType:
    """
    Create a plot from a BackTest object.

    This function handles conversion from BackTest to flat DataFrames and
    instantiates the appropriate plot class.

    Parameters
    ----------
    plot_id : str
        The unique identifier of the plot to create
    backtest : BackTest
        The backtest object containing forecast and observation data

    Returns
    -------
    ChartType
        The generated Altair chart

    Raises
    ------
    ValueError
        If the plot_id is not found in the registry
    """
    from chap_core.assessment.evaluation import Evaluation

    plot_cls = get_backtest_plot(plot_id)
    if plot_cls is None:
        available = ", ".join(_backtest_plots_registry.keys())
        raise ValueError(f"Unknown plot type: {plot_id}. Available: {available}")

    # Convert BackTest to flat DataFrames via Evaluation
    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()

    # Get flat DataFrames - FlatObserved/FlatForecasts are already DataFrames
    observations_df: pd.DataFrame = flat_data.observations  # type: ignore[assignment]
    forecasts_df: pd.DataFrame = flat_data.forecasts  # type: ignore[assignment]

    # Get historical observations if the plot needs them
    historical_df: pd.DataFrame | None = None
    if plot_cls.needs_historical and flat_data.historical_observations is not None:
        historical_df = flat_data.historical_observations  # type: ignore[assignment]

    # Create plot instance and generate chart
    plotter = plot_cls()
    return plotter.plot(observations_df, forecasts_df, historical_df)


def create_plot_from_evaluation(plot_id: str, evaluation) -> ChartType:
    """
    Create a plot from an Evaluation object.

    This function handles conversion from Evaluation to flat DataFrames and
    instantiates the appropriate plot class.

    Parameters
    ----------
    plot_id : str
        The unique identifier of the plot to create
    evaluation : Evaluation
        The evaluation object containing forecast, observation, and historical data

    Returns
    -------
    ChartType
        The generated Altair chart

    Raises
    ------
    ValueError
        If the plot_id is not found in the registry
    """
    plot_cls = get_backtest_plot(plot_id)
    if plot_cls is None:
        available = ", ".join(_backtest_plots_registry.keys())
        raise ValueError(f"Unknown plot type: {plot_id}. Available: {available}")

    flat_data = evaluation.to_flat()

    # Get flat DataFrames - FlatObserved/FlatForecasts are already DataFrames
    observations_df: pd.DataFrame = flat_data.observations  # type: ignore[assignment]
    forecasts_df: pd.DataFrame = flat_data.forecasts  # type: ignore[assignment]

    # Get historical observations if the plot needs them
    historical_df: pd.DataFrame | None = None
    if plot_cls.needs_historical and flat_data.historical_observations is not None:
        historical_df = flat_data.historical_observations  # type: ignore[assignment]

    # Create plot instance and generate chart
    plotter = plot_cls()
    return plotter.plot(observations_df, forecasts_df, historical_df)


# Import plot modules to trigger registration
# Each plot file uses the @backtest_plot decorator which registers it
def _discover_plots():
    """Import all plot modules to trigger decorator registration."""
    from chap_core.assessment.backtest_plots import (  # noqa: F401
        evaluation_plot,
        metrics_dashboard,
        sample_bias_plot,
    )


_discover_plots()
