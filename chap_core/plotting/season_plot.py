import numpy as np
import pandas as pd
import altair as alt
from .dataset_plot import DatasetPlot

alt.renderers.default = "browser"


class SeasonPlot(DatasetPlot):
    def data(self) -> pd.DataFrame:
        df = self._df.copy()
        df["log1p"] = np.log1p(df["disease_cases"])
        df["time_period"] = pd.to_datetime(df["time_period"])
        df["month"] = df["time_period"].dt.month - 1
        df["year"] = df["time_period"].dt.year
        means = ((month, group["log1p"].mean()) for month, group in df.groupby("month"))
        min_month, val = min(means, key=lambda x: x[1])
        assert df["month"].max() == 11
        offset_month = df["month"] - min_month
        df["seasonal_month"] = offset_month % 12
        df["season_idx"] = df["year"] + offset_month // 12
        # Create season_idx (season index based on years from start)
        df["season_idx"] = df["season_idx"] - df["season_idx"].min()
        return df

    def plot(self) -> alt.FacetChart:
        df = self.data()
        return (
            alt.Chart(df)
            .mark_line(point=False, strokeWidth=2)
            .encode(
                x=alt.X("seasonal_month:O", title="Month"),
                y=alt.Y("log1p:Q", title="Log1p Disease Cases"),
                color=alt.Color("season_idx:N", title="Season Year"),
            )
            .facet(facet=alt.Facet("location:N", title="Location"), columns=3)
        )


class SeasonCorrelationPlot(DatasetPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._season_plot = SeasonPlot(*args, **kwargs)

    def data(self) -> pd.DataFrame:
        df = self._season_plot.data()
        season_stats = df.groupby(["location", "season_idx"])["log1p"].agg(["mean", "std", "max"]).reset_index()
        season_stats.columns = ["location", "season_idx", "season_mean", "season_std", "season_max"]
        df = df.merge(season_stats, on=["location", "season_idx"], how="left")
        return df

    def plot(self):
        df = self.data()
        return (
            alt.Chart(df)
            .mark_point(filled=True, size=100)
            .encode(
                x=alt.X("mean_temperature:Q", title="Predictor: Mean Temperature", scale=alt.Scale(zero=False)),
                y=alt.Y("season_max:Q", title="Season Max (Log1p Disease Cases)", scale=alt.Scale(zero=False)),
                color=alt.Color("seasonal_month:N", title="Seasonal Month"),
                tooltip=["location:N", "season_idx:N", "season_mean:Q", "season_std:Q", "mean_temperature:Q"],
            )
            .facet(
                row=alt.Row("location:N", title="Location"),
                column=alt.Column("seasonal_month:O", title="Seasonal Month"),
            )
            .resolve_scale(y="independent", x="independent")
        )


class SeasonCorrelationBarPlot(SeasonCorrelationPlot):
    feature_name = "mean_temperature"  # Example feature to correlate with season_mean

    def data(self) -> pd.DataFrame:
        df = super().data()
        last_months_subset = df[df["seasonal_month"] >= 9].copy()
        last_months_subset["seasonal_month"] -= 12
        last_months_subset["season_idx"] += 1
        df = pd.concat([df, last_months_subset], ignore_index=True)

        # Calculate correlation coefficient between season_mean and mean_temperature for each season_idx and location
        correlations = []
        for (location, seasonal_month), group in df.groupby(["location", "seasonal_month"]):
            for feature_name in self._get_feature_names():
                for outcome in ["max", "mean", "std"]:
                    # Suppress RuntimeWarning for invalid values (e.g., when std is 0)
                    with np.errstate(invalid="ignore"):
                        corr = group[f"season_{outcome}"].corr(group[feature_name])
                    correlations.append(
                        {
                            "location": location,
                            "seasonal_month": seasonal_month,
                            "correlation": corr,
                            "feature": feature_name,
                            "outcome": outcome,
                            "combination": f"{feature_name}_vs_season_{outcome}",
                        }
                    )

        return pd.DataFrame(correlations)

    def plot(self) -> alt.FacetChart:
        df = self.data()
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("seasonal_month:O", title="Seasonal Month"),
                y=alt.Y("correlation:Q", title="Correlation (Season Max vs Temperature)"),
                color=alt.Color(
                    "correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="Correlation"
                ),
                tooltip=["location:N", "seasonal_month:O", "correlation:Q"],
            )
            .facet(
                column=alt.Column("location:N", title="Location"),
                row=alt.Row("combination:N", title="Feature vs Outcome"),
            )
            .properties(
                title={
                    "text": "Seasonal Correlation Analysis",
                    "subtitle": "Correlation between season maximum disease cases and mean temperature by seasonal month and location. Red bars indicate negative correlation, blue bars indicate positive correlation.",
                }
            )
        )


def test_season_plot(df: pd.DataFrame):
    plot = SeasonPlot(df)
    data = plot.data()

    print(data)
    assert "seasonal_month" in data.columns
    chart = plot.plot()
    chart.save("season_plot.html")
    chart.save("season_plot.png")


def test_season_correlation_plot(df: pd.DataFrame):
    plot = SeasonCorrelationPlot(df)
    data = plot.data()
    print(data)
    assert "season_mean" in data.columns
    assert "season_std" in data.columns
    chart = plot.plot()
    chart.save("season_correlation_plot.html")
    chart.save("season_correlation_plot.png")


def test_season_correlation_bar_plot(df: pd.DataFrame):
    plot = SeasonCorrelationBarPlot(df)
    data = plot.data()
    print(data)
    assert "correlation" in data.columns
    assert "location" in data.columns
    chart = plot.plot()
    chart.save("season_correlation_bar_plot.html")
    chart.save("season_correlation_bar_plot.png")
