"""
Metric plot plugin system.

This module provides a clean interface for creating custom metric visualizations.
Each plot is a single class that receives a pre-computed metric DataFrame and
returns an Altair chart.

Example usage:
    from chap_core.assessment.metric_plots import metric_plot, MetricPlotBase
    import altair as alt

    @metric_plot(
        plot_id="my_custom_plot",
        name="My Custom Plot",
        description="A custom visualization of metric values.",
    )
    class MyCustomPlot(MetricPlotBase):
        def plot_from_df(self, title: str = "") -> alt.Chart:
            # self._metric_data has columns: location, time_period, horizon_distance, metric
            ...
            return chart

Data schema:
    metric_data (pd.DataFrame):
        - location: str - Location identifier
        - time_period: str - Time period (e.g., "2024-01" or "2024W01")
        - horizon_distance: int - How many periods ahead this forecast is
        - metric: float - Computed metric value (e.g., RMSE, CRPS)
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import altair as alt

if TYPE_CHECKING:
    import pandas as pd

# Global registry for metric plots
_metric_plots_registry: dict[str, type[MetricPlotBase]] = {}


class MetricPlotBase(abc.ABC):
    """
    Base class for metric visualizations.

    Subclasses must implement `plot_from_df` which receives a pre-computed
    metric DataFrame and returns an Altair chart.

    Attributes:
        id: Unique identifier (set by decorator)
        name: Display name (set by decorator)
        description: Description of what the plot shows (set by decorator)
    """

    id: str = ""
    name: str = ""
    description: str = ""

    def __init__(self, metric_data: pd.DataFrame, geojson: dict | None = None):
        self._metric_data = metric_data

    def plot(self, title: str = "") -> alt.Chart:
        return self.plot_from_df(title=title)

    @abc.abstractmethod
    def plot_from_df(self, title: str = "") -> alt.Chart:
        """
        Generate the visualization from pre-computed metric data.

        Parameters
        ----------
        title : str
            Chart title.

        Returns
        -------
        alt.Chart
            Altair chart specification.
        """

    def plot_spec(self) -> dict:
        return self.plot().to_dict(format="vega")


def metric_plot(
    plot_id: str,
    name: str,
    description: str = "",
):
    """
    Decorator to register a metric plot class.

    Parameters
    ----------
    plot_id : str
        Unique identifier for the plot (used in URLs/APIs).
    name : str
        Human-readable display name.
    description : str, optional
        Description of what the plot shows.

    Example
    -------
    @metric_plot(
        plot_id="metric_by_horizon_mean",
        name="Horizon Plot",
        description="Mean metric by forecast horizon",
    )
    class MyPlot(MetricPlotBase):
        def plot_from_df(self, title: str = "") -> alt.Chart:
            ...
    """

    def decorator(cls: type[MetricPlotBase]) -> type[MetricPlotBase]:
        if not issubclass(cls, MetricPlotBase):
            raise TypeError(f"{cls.__name__} must inherit from MetricPlotBase")

        cls.id = plot_id
        cls.name = name
        cls.description = description

        _metric_plots_registry[plot_id] = cls
        return cls

    return decorator


def get_metric_plots_registry() -> dict[str, type[MetricPlotBase]]:
    """Return a copy of the registry mapping plot IDs to plot classes."""
    return _metric_plots_registry.copy()


def list_metric_plots() -> list[dict]:
    """
    List all registered metric plots with their metadata.

    Returns
    -------
    list[dict]
        List of dicts with id, name, description for each plot.
    """
    return [{"id": cls.id, "name": cls.name, "description": cls.description} for cls in _metric_plots_registry.values()]


def _discover_plots() -> None:
    """Import all plot modules to trigger decorator registration."""
    from chap_core.assessment.metric_plots import (
        horizon_location_mean,
        horizon_mean,
        horizon_sum,
        metric_map,
        regional_distribution,
        time_period_location_mean,
        time_period_mean,
        time_period_sum,
    )


_discover_plots()
