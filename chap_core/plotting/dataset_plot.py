from abc import ABC, abstractmethod
import altair as alt
import numpy as np
import pandas as pd

from altair import HConcatChart

from chap_core.spatio_temporal_data.converters import dataset_model_to_dataset

alt.data_transformers.enable("default")
alt.renderers.enable("browser")


# alt.renderers.enable('notebook')


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
    def __init__(self, df: pd.DataFrame, geojson=None):
        # df['ideal_temperature'] = temperature_transform(df['mean_temperature'])
        self._df = df
        self._geojson = geojson

    @classmethod
    def from_dataset_model(cls, dataset_model):
        ds = dataset_model_to_dataset(dataset_model)
        df = ds.to_pandas()
        geojson = ds.polygons
        return cls.from_pandas(df, geojson=geojson)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, geojson=None):
        df = df.copy()
        df["time_period"] = df["time_period"].astype(str)
        return cls(df, geojson=geojson)

    def _get_feature_names(self) -> list:
        return [name for name in self._get_colnames() if name not in ("log1p", "log1p", "population")]

    def _get_colnames(self) -> filter:
        colnames = filter(
            lambda name: name not in ("disease_cases", "location", "time_period") and not name.startswith("Unnamed"),
            self._df.columns,
        )
        colnames = filter(
            lambda name: self._df[name].dtype.name in ("float64", "int64", "bool", "int32", "float32"), colnames
        )
        print(self._df.columns)
        colnames = list(colnames)
        print(colnames)
        return colnames

    def plot_spec(self):
        return self.plot().to_dict(format="vega")

    @abstractmethod
    def plot(self) -> alt.Chart: ...

    @abstractmethod
    def data(self): ...


class StandardizedFeaturePlot(DatasetPlot):
    """
    This plot shows standardized(zero mean, unit variance) features over time for different locations.
    It includes a log1p transformation of the disease incidence rate (disease_cases/population)
    This shows how different features correlate over time and location.
    """

    def _standardize(self, col: np.array) -> np.array:
        # Handle NaN values properly
        mean_val = np.nanmean(col)
        std_val = np.nanstd(col)
        if std_val == 0:
            return col - mean_val  # Return zero-centered values when std is 0
        return (col - mean_val) / std_val

    def data(self) -> pd.DataFrame:
        df = self._df.copy()
        colnames = list(self._get_colnames())
        base_df = df[["time_period", "location"]].copy()

        # Add log1p of disease incidence rate if population column exists
        if "population" in df.columns:
            df["log1p"] = np.log1p(df["disease_cases"] / df["population"])
            colnames.append("log1p")
        else:
            # Fallback to just log1p of disease cases
            df["log1p"] = np.log1p(df["disease_cases"])
            colnames.append("log1p")

        dfs = []
        for colname in colnames:
            if colname in df.columns:
                new_df = base_df.copy()
                new_df["value"] = self._standardize(df[colname].values)
                new_df["feature"] = colname
                dfs.append(new_df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            # Return empty dataframe with correct structure
            return pd.DataFrame(columns=["time_period", "location", "value", "feature"])

    def plot(self) -> HConcatChart:
        data = self.data()

        # Filter data based on selected features if specified
        # Convert time_period to proper datetime format
        data["date"] = pd.to_datetime(data["time_period"] + "-01")

        checkbox_selection = alt.selection_point(fields=["feature"], toggle="true")

        # Create legend that acts as checkboxes
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

        # Main chart with filtering
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
        return (
            alt.hconcat(legend_chart, main_chart)
            .resolve_legend(color="independent")
            .properties(title="Multiple Feature Selection (Click legend items to toggle)")
        )


def test_standardized_feature_plot(df: pd.DataFrame):
    df["ideal_temperature"] = (df["mean_temperature"] > 25) & (
        df["mean_temperature"] <= 30
    )  # Assuming mean_temperature is the predictor
    df["ideal_temperature"] = df["ideal_temperature"].astype(int)
    # Convert boolean to int for plotting
    plotter = StandardizedFeaturePlot(df)

    data = plotter.data()
    print(data.head())
    print(f"Data shape: {data.shape}")
    print(f"Features: {data['feature'].unique()}")
    assert "value" in data.columns
    assert "feature" in data.columns
    assert "location" in data.columns
    assert "time_period" in data.columns

    chart = plotter.plot()
    chart.save("standardized_feature_plot.html")
    chart.save("standardized_feature_plot.png")
    print("Chart saved to standardized_feature_plot.html and standardized_feature_plot.png")


def test_temperature_transform():
    temps = np.arange(35)
    transformed = temperature_transform(temps)
    df = pd.DataFrame({"mean_temperature": temps, "ideal_temperature": transformed})
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x="mean_temperature", y="ideal_temperature")
        .properties(title="Temperature Transformation")
    )
    chart.save("temperature_transform.html")
    chart.save("temperature_transform.png")
