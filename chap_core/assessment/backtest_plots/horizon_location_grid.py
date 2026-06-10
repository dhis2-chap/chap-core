"""
Horizon x Location grid plot for backtests.

Shows a grid of forecast cells (locations as rows, horizons as columns).
Each cell contains error bands + observed values on top, and per-metric
line charts stacked below.
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import ChartType, FacetDimension, FacetedBacktestPlot, backtest_plot
from chap_core.assessment.backtest_plots.db_dimensions import HorizonDistanceDimension, LocationDimension
from chap_core.assessment.backtest_plots.evaluation_plot import _compute_quantiles_from_forecasts
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.assessment.metrics.crps import CRPSLog1pMetric, CRPSMetric
from chap_core.assessment.metrics.outbreak_detection import OutbreakAccuracyMetric
from chap_core.assessment.metrics.winkler_score import WinklerScore10_90Log1pMetric, WinklerScore10_90Metric
from chap_core.plotting.backtest_plot import clean_time

CELL_WIDTH = 250
CELL_HEIGHT = 150
METRIC_HEIGHT = 60


def _empty_placeholder(width: int, height: int, title: str = "") -> ChartType:
    """Return an empty chart placeholder that won't trip vegafusion's type checks."""
    return (  # type: ignore[no-any-return]
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_point(opacity=0)
        .encode(x=alt.value(0), y=alt.value(0))
        .properties(width=width, height=height, title=title)
    )


def _build_forecast_cell(
    forecast_quantiles: pd.DataFrame,
    obs_df: pd.DataFrame,
    location: str,
    horizon: int,
) -> ChartType:
    """Build the forecast intervals + observed sub-chart for one cell."""
    fq = forecast_quantiles[
        (forecast_quantiles["location"] == location) & (forecast_quantiles["horizon_distance"] == horizon)
    ].copy()
    obs = obs_df[obs_df["location"] == location].copy()

    # Only keep observation time_periods that appear in this horizon's forecasts
    obs = obs[obs["time_period"].isin(fq["time_period"])]

    if fq.empty:
        return _empty_placeholder(CELL_WIDTH, CELL_HEIGHT, f"{location} | horizon {horizon}")

    base_forecast = alt.Chart(fq)

    error_10_90 = base_forecast.mark_area(opacity=0.3, color="steelblue").encode(
        x=alt.X("time_period:T", title=None),
        y=alt.Y("q_10:Q", scale=alt.Scale(zero=False), title="Cases"),
        y2="q_90:Q",
    )
    error_25_75 = base_forecast.mark_area(opacity=0.5, color="steelblue").encode(
        x=alt.X("time_period:T", title=None),
        y=alt.Y("q_25:Q", scale=alt.Scale(zero=False), title="Cases"),
        y2="q_75:Q",
    )
    median_line = base_forecast.mark_line(color="steelblue").encode(
        x="time_period:T",
        y="q_50:Q",
    )

    layers = error_10_90 + error_25_75 + median_line

    # Only add observations layer if there is data; an empty DataFrame with
    # object-typed columns causes vegafusion to fail on temporal encoding.
    if not obs.empty:
        obs_line = (
            alt.Chart(obs)
            .mark_line(color="orange")
            .encode(
                x="time_period:T",
                y=alt.Y("disease_cases:Q", scale=alt.Scale(zero=False)),
            )
        )
        layers = layers + obs_line

    return layers.properties(  # type: ignore[no-any-return]
        width=CELL_WIDTH, height=CELL_HEIGHT, title=f"{location} | horizon {horizon}"
    )


def _build_metric_chart(
    metric_df: pd.DataFrame,
    location: str,
    horizon: int,
    metric_name: str,
) -> ChartType:
    """Build a small line chart for a single metric in one cell."""
    subset = metric_df[(metric_df["location"] == location) & (metric_df["horizon_distance"] == horizon)].copy()

    if subset.empty:
        return _empty_placeholder(CELL_WIDTH, METRIC_HEIGHT)

    return (  # type: ignore[no-any-return]
        alt.Chart(subset)
        .mark_line(point=True)
        .encode(
            x=alt.X("time_period:T", title=None),
            y=alt.Y("metric:Q", title=metric_name, scale=alt.Scale(zero=False)),
        )
        .properties(width=CELL_WIDTH, height=METRIC_HEIGHT)
    )


def _build_summary_table(
    metric_dfs: list[tuple[str, pd.DataFrame]],
    horizons: list[int],
    *,
    location: str | None = None,
) -> ChartType:
    """Build a text table showing mean metric values.

    If *location* is given, aggregate only that location's data (not used currently).
    When *location* is None, aggregate across all locations.
    """
    rows = []
    for metric_name, mdf in metric_dfs:
        for hz in horizons:
            subset = mdf[mdf["horizon_distance"] == hz]
            if location is not None:
                subset = subset[subset["location"] == location]
            mean_val = subset["metric"].mean() if not subset.empty else float("nan")
            rows.append({"metric": metric_name, "horizon": f"hz {hz}", "value": f"{mean_val:.2f}"})
        # Overall (all horizons)
        subset = mdf if location is None else mdf[mdf["location"] == location]
        mean_val = subset["metric"].mean() if not subset.empty else float("nan")
        rows.append({"metric": metric_name, "horizon": "overall", "value": f"{mean_val:.2f}"})

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_placeholder(CELL_WIDTH, 60)

    horizon_order = [f"hz {hz}" for hz in horizons] + ["overall"]

    return (  # type: ignore[no-any-return]
        alt.Chart(df)
        .mark_text(align="center", baseline="middle", fontSize=12)
        .encode(
            x=alt.X("horizon:N", sort=horizon_order, title=None),
            y=alt.Y("metric:N", title=None),
            text="value:N",
        )
        .properties(width=CELL_WIDTH * len(horizons), height=len(metric_dfs) * 25 + 20)
    )


def _metric_instances(historical_observations: pd.DataFrame | None) -> list:
    """The metrics shown in each grid cell, in display order."""
    return [
        CRPSMetric(historical_observations=historical_observations),
        CRPSLog1pMetric(historical_observations=historical_observations),
        WinklerScore10_90Metric(historical_observations=historical_observations),
        WinklerScore10_90Log1pMetric(historical_observations=historical_observations),
        OutbreakAccuracyMetric(historical_observations=historical_observations),
    ]


def _metric_dfs_from_preprocessed(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Recover the per-metric frames from the packed preprocessed frame, in order."""
    metric_rows = df[df["role"] == "metric"]
    return [(str(name), group) for name, group in metric_rows.groupby("metric_name", sort=False)]


@backtest_plot(
    plot_id="horizon_location_grid",
    name="Forecast Grid (Locations x Horizons)",
    description="Grid of forecast intervals and metrics across locations and forecast horizons.",
    needs_historical=True,
)
class HorizonLocationGridPlot(FacetedBacktestPlot):
    """Grid plot with locations as rows and forecast horizons as columns.

    Each cell is a composite (forecast band + observed + per-metric line charts),
    so the all-cells view (`get_full_plot`) keeps the hand-built grid, while
    `facet_coords` / `get_subplot` expose a single location x horizon cell for the
    database-backed faceting workflow.
    """

    facet_dimensions: list[FacetDimension] = [
        HorizonDistanceDimension(field_name="horizon_distance:O", display_name="Horizon Distance"),
        LocationDimension(field_name="location:N", display_name="Location"),
    ]

    def _preprocess(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Pack forecast quantiles, observed values and metric series into one frame.

        Each row carries `location` + `horizon_distance` (observations replicated
        across horizons) so the generic coordinate filter can slice to one cell, plus
        a `role` discriminator the cell renderer splits back out.
        """
        flat_obs = FlatObserved(observations)
        flat_fc = FlatForecasts(forecasts)
        horizons = sorted(forecasts["horizon_distance"].unique())

        forecast_quantiles = _compute_quantiles_from_forecasts(forecasts)
        forecast_quantiles["time_period"] = forecast_quantiles["time_period"].apply(clean_time)
        forecast_quantiles["role"] = "forecast"

        obs_df = observations.copy()
        obs_df["time_period"] = obs_df["time_period"].apply(clean_time)
        # Observations have no horizon; replicate across horizons so the cell filter keeps them.
        obs_df = obs_df.merge(pd.DataFrame({"horizon_distance": horizons}), how="cross")
        obs_df["role"] = "observed"

        frames: list[pd.DataFrame] = [forecast_quantiles, obs_df]
        for m in _metric_instances(historical_observations):
            if not m.is_applicable(flat_obs):
                continue
            detailed = m.get_detailed_metric(flat_obs, flat_fc)
            detailed["time_period"] = detailed["time_period"].apply(clean_time)
            detailed["role"] = "metric"
            detailed["metric_name"] = m.get_name()
            frames.append(detailed)

        return pd.concat(frames, ignore_index=True)

    def _plot(self, df: pd.DataFrame) -> ChartType:
        """Render one location x horizon cell from the packed frame."""
        if df.empty:
            return _empty_placeholder(CELL_WIDTH, CELL_HEIGHT)

        forecast_rows = df[df["role"] == "forecast"]
        observed_rows = df[df["role"] == "observed"]
        key_src = forecast_rows if not forecast_rows.empty else df
        location = key_src["location"].iloc[0]
        horizon = int(key_src["horizon_distance"].iloc[0])

        cell_parts: list[ChartType] = [_build_forecast_cell(forecast_rows, observed_rows, location, horizon)]
        for metric_name, mdf in _metric_dfs_from_preprocessed(df):
            cell_parts.append(_build_metric_chart(mdf, location, horizon, metric_name))

        return alt.vconcat(*cell_parts).properties(spacing=2)  # type: ignore[no-any-return]

    def get_full_plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        """The full grid: composite cells (locations as rows, horizons as columns) plus a summary table.

        Overrides the base native-faceting implementation, which cannot tile the
        composite (vconcat) cells this plot renders.
        """
        df = self._preprocess(observations, forecasts, historical_observations)
        locations = sorted(forecasts["location"].unique())
        horizons = sorted(forecasts["horizon_distance"].unique())

        location_rows: list[ChartType] = []
        for loc in locations:
            horizon_cells = [
                self._plot(df[(df["location"] == loc) & (df["horizon_distance"] == hz)]) for hz in horizons
            ]
            location_rows.append(alt.hconcat(*horizon_cells).properties(spacing=10))

        metric_dfs = _metric_dfs_from_preprocessed(df)
        if metric_dfs:
            location_rows.append(_build_summary_table(metric_dfs, horizons))

        return alt.vconcat(*location_rows).properties(  # type: ignore[no-any-return]
            spacing=15,
            title="Forecast Grid: Locations x Horizons",
        )
