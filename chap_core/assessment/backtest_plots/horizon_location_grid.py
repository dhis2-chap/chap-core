"""
Horizon x Location grid plot for backtests.

Shows a grid of forecast cells (locations as rows, horizons as columns).
Each cell contains error bands + observed values on top, and per-metric
line charts stacked below.
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot
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


@backtest_plot(
    plot_id="horizon_location_grid",
    name="Forecast Grid (Locations x Horizons)",
    description="Grid of forecast intervals and metrics across locations and forecast horizons.",
    needs_historical=True,
)
class HorizonLocationGridPlot(BacktestPlotBase):
    """Grid plot with locations as rows and forecast horizons as columns."""

    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        flat_obs = FlatObserved(observations)
        flat_fc = FlatForecasts(forecasts)

        # Compute quantiles for forecast cells
        forecast_quantiles = _compute_quantiles_from_forecasts(forecasts)
        forecast_quantiles["time_period"] = forecast_quantiles["time_period"].apply(clean_time)

        obs_df = observations.copy()
        obs_df["time_period"] = obs_df["time_period"].apply(clean_time)

        locations = sorted(forecasts["location"].unique())
        horizons = sorted(forecasts["horizon_distance"].unique())

        # Compute metrics
        metric_instances = [
            CRPSMetric(historical_observations=historical_observations),
            CRPSLog1pMetric(historical_observations=historical_observations),
            WinklerScore10_90Metric(historical_observations=historical_observations),
            WinklerScore10_90Log1pMetric(historical_observations=historical_observations),
            OutbreakAccuracyMetric(historical_observations=historical_observations),
        ]

        metric_dfs: list[tuple[str, pd.DataFrame]] = []
        for m in metric_instances:
            if not m.is_applicable(flat_obs):
                continue
            detailed = m.get_detailed_metric(flat_obs, flat_fc)
            detailed["time_period"] = detailed["time_period"].apply(clean_time)
            metric_dfs.append((m.get_name(), detailed))

        # Build grid: rows = locations, columns = horizons
        location_rows: list[ChartType] = []
        for loc in locations:
            horizon_cells = []
            for hz in horizons:
                cell_parts: list[ChartType] = [_build_forecast_cell(forecast_quantiles, obs_df, loc, hz)]
                for metric_name, mdf in metric_dfs:
                    cell_parts.append(_build_metric_chart(mdf, loc, hz, metric_name))
                horizon_cells.append(alt.vconcat(*cell_parts).properties(spacing=2))
            location_rows.append(alt.hconcat(*horizon_cells).properties(spacing=10))

        # Aggregated summary tables
        if metric_dfs:
            summary = _build_summary_table(metric_dfs, horizons)
            location_rows.append(summary)

        return alt.vconcat(*location_rows).properties(  # type: ignore[return-value]
            spacing=15,
            title="Forecast Grid: Locations x Horizons",
        )
