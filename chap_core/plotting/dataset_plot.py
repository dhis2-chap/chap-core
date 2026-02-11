from abc import ABC, abstractmethod
from typing import Optional, Union

import altair as alt
import numpy as np
import pandas as pd
from altair import HConcatChart
from pydantic import BaseModel

from chap_core.spatio_temporal_data.converters import dataset_model_to_dataset

alt.data_transformers.enable("vegafusion")

ChartType = Union[alt.Chart, alt.VConcatChart, alt.FacetChart, alt.LayerChart, alt.HConcatChart]

# Global registry for dataset plots
_dataset_plots_registry: dict[str, type["DatasetPlot"]] = {}


def temperature_transform(x):
    """
    Transforming function f(x) that:
    - Is low (around 0) until 15
    - Starts growing with highest growth around 25
    - Plateaus at 30

    Uses a sigmoid-like function with shifted center and scaling
    """
    return 1 / (1 + np.exp(-0.5 * (x - 25)))


class DatasetPlot(ABC):
    id: str = ""
    name: str = ""
    description: str = ""

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["time_period"] = df["time_period"].astype(str)
        return df

    def plot_from_dataset_model(self, dataset_model) -> ChartType:
        """Create a plot from a DB DataSet model."""
        ds = dataset_model_to_dataset(dataset_model)
        return self.plot_from_dataset(ds)

    def plot_from_dataset(self, dataset) -> ChartType:
        """Create a plot from an in-memory DataSet object."""
        df = self._prepare_df(dataset.to_pandas())
        geojson = dataset.polygons
        if isinstance(geojson, BaseModel):
            geojson = geojson.model_dump()
        return self.plot(df, geojson=geojson)

    def _get_feature_names(self, df: pd.DataFrame) -> list:
        return [name for name in self._get_colnames(df) if name not in ("log1p", "log1p", "population")]

    def _get_colnames(self, df: pd.DataFrame) -> list[str]:
        colnames = list(
            filter(
                lambda name: df[name].dtype.name in ("float64", "int64", "bool", "int32", "float32"),
                filter(
                    lambda name: name not in ("disease_cases", "location", "time_period")
                    and not name.startswith("Unnamed"),
                    df.columns,
                ),
            )
        )
        return colnames

    def plot_spec(self, df: pd.DataFrame, geojson=None):
        return self.plot(df, geojson=geojson).to_dict(format="vega")

    @abstractmethod
    def plot(self, df: pd.DataFrame, geojson=None) -> ChartType: ...


def dataset_plot(id: str, name: str, description: str = ""):
    """Decorator to register a dataset plot class."""

    def decorator(cls: type[DatasetPlot]) -> type[DatasetPlot]:
        cls.id = id
        cls.name = name
        cls.description = description
        _dataset_plots_registry[id] = cls
        return cls

    return decorator


def get_dataset_plots_registry() -> dict[str, type[DatasetPlot]]:
    """Get the registry of all registered dataset plots."""
    return _dataset_plots_registry.copy()


def get_dataset_plot(plot_id: str) -> Optional[type[DatasetPlot]]:
    """Get a specific dataset plot class by ID."""
    return _dataset_plots_registry.get(plot_id)


def list_dataset_plots() -> list[dict]:
    """List all registered dataset plots with their metadata."""
    return [
        {"id": cls.id, "name": cls.name, "description": cls.description} for cls in _dataset_plots_registry.values()
    ]


def create_plot_from_dataset(plot_id: str, dataset):
    """Create a plot spec from a dataset model using the registry."""
    plot_cls = get_dataset_plot(plot_id)
    if plot_cls is None:
        available = ", ".join(_dataset_plots_registry.keys())
        raise ValueError(f"Unknown plot type: {plot_id}. Available: {available}")
    plotter = plot_cls()
    chart = plotter.plot_from_dataset_model(dataset)
    return chart.to_dict(format="vega")


@dataset_plot(
    id="disease-cases-map",
    name="Disease Cases Map",
    description="Choropleth map showing mean disease cases or incidence rate by location.",
)
class DiseaseCasesMap(DatasetPlot):
    def plot(self, df: pd.DataFrame, geojson=None) -> alt.Chart:  # type: ignore[override]
        plot_variable = "disease_cases"
        df = df.copy()
        if "population" in df.columns:
            df["incidence_rate"] = df["disease_cases"] / df["population"]
            plot_variable = "incidence_rate"
        agg = df.groupby("location").agg({plot_variable: "mean"}).reset_index()

        if geojson is None:
            raise ValueError("GeoJSON data is required for DiseaseCasesMap")

        geojson_data = alt.Data(values=geojson["features"])

        chart = (
            alt.Chart(geojson_data)
            .mark_geoshape(stroke="white", strokeWidth=0.5)
            .transform_lookup(lookup="id", from_=alt.LookupData(agg, "location", [plot_variable]))
            .encode(
                color=alt.Color(
                    f"{plot_variable}:Q",
                    scale=alt.Scale(scheme="oranges"),
                    legend=alt.Legend(
                        title="Mean Disease Cases" if plot_variable == "disease_cases" else "Mean Incidence Rate"
                    ),
                ),
                tooltip=[
                    alt.Tooltip("id:N", title="Location"),
                    alt.Tooltip(
                        f"{plot_variable}:Q",
                        title="Mean Disease Cases" if plot_variable == "disease_cases" else "Mean Incidence Rate",
                        format=".2f",
                    ),
                ],
            )
            .project(type="equirectangular")
            .properties(
                width=600,
                height=400,
                title="Disease Cases Map" if plot_variable == "disease_cases" else "Disease Incidence Rate Map",
            )
        )

        return chart  # type: ignore[no-any-return]


@dataset_plot(
    id="standardized-feature-plot",
    name="Standardized Feature Plot",
    description="Standardized features over time for different locations with interactive selection.",
)
class StandardizedFeaturePlot(DatasetPlot):
    """
    This plot shows standardized(zero mean, unit variance) features over time for different locations.
    It includes a log1p transformation of the disease incidence rate (disease_cases/population)
    This shows how different features correlate over time and location.
    """

    def _standardize(self, col: np.ndarray) -> np.ndarray:
        # Handle NaN values properly
        mean_val = np.nanmean(col)
        std_val = np.nanstd(col)
        if std_val == 0:
            return col - mean_val  # type: ignore[no-any-return]
        return (col - mean_val) / std_val  # type: ignore[no-any-return]

    def data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        colnames = list(self._get_colnames(df))
        base_df = df[["time_period", "location"]].copy()

        if "population" in df.columns:
            df["log1p"] = np.log1p(df["disease_cases"] / df["population"])
            colnames.append("log1p")
        else:
            df["log1p"] = np.log1p(df["disease_cases"])
            colnames.append("log1p")

        dfs = []
        for colname in colnames:
            if colname in df.columns:
                new_df = base_df.copy()
                new_df["value"] = self._standardize(df[colname].values)  # type: ignore[arg-type]
                new_df["feature"] = colname
                dfs.append(new_df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame(columns=["time_period", "location", "value", "feature"])

    def plot(self, df: pd.DataFrame, geojson=None) -> HConcatChart:  # type: ignore[override]
        data = self.data(df)

        data["date"] = pd.to_datetime(data["time_period"] + "-01")

        checkbox_selection = alt.selection_point(fields=["feature"], toggle="true")

        legend_chart = (
            alt.Chart(data)
            .mark_circle(size=100)
            .add_params(checkbox_selection)
            .encode(
                y=alt.Y("feature:N", axis=alt.Axis(orient="right", title="Select Features")),
                color=alt.condition(checkbox_selection, alt.Color("feature:N", legend=None), alt.value("lightgray")),
                tooltip=["feature:N"],
            )
            .properties(width=100, title="Click to select/deselect")
        )

        main_chart = (
            alt.Chart(data)
            .add_params(alt.selection_interval(bind="scales", encodings=["x"]))
            .mark_line(point=False, strokeWidth=2)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Standardized Value"),
                color=alt.Color("feature:N", legend=alt.Legend(title="Feature")),
                opacity=alt.condition(checkbox_selection, alt.value(1.0), alt.value(0.1)),
                tooltip=["date:T", "feature:N", "value:Q", "location:N"],
            )
            .facet(facet=alt.Facet("location:N", title="Location"), columns=3)
            .resolve_scale(y="shared")
        )
        return (  # type: ignore[no-any-return]
            alt.hconcat(legend_chart, main_chart)
            .resolve_legend(color="independent")
            .properties(title="Multiple Feature Selection (Click legend items to toggle)")
        )


def _discover_plots():
    """Import plot modules to trigger decorator registration."""
    from chap_core.plotting import season_plot  # noqa: F401


_discover_plots()
