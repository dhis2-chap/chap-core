"""
Covariate importance radar chart using Spearman correlation.

Shows a radar (spider) chart of absolute Spearman rank correlations between
each covariate and the target variable (disease_cases). Each axis represents
a covariate, and the radial distance shows correlation strength (0-1).

Uses a raw Vega spec since Altair/Vega-Lite do not support radial layouts.

Inspired by the Uganda Nutrition Early Warning tool's covariate importance
chart (CLIM-538).
"""

import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from chap_core.assessment.backtest_plots import BacktestPlotBase, PlotResult, backtest_plot


def _empty_chart(message: str) -> PlotResult:
    """Return a minimal Vega spec with a centered text message."""
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": 400,
        "height": 100,
        "marks": [
            {
                "type": "text",
                "encode": {
                    "enter": {
                        "x": {"value": 200},
                        "y": {"value": 50},
                        "text": {"value": message},
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "fontSize": {"value": 14},
                        "fill": {"value": "#666"},
                    }
                },
            }
        ],
    }


def _build_radar_spec(corr_df: pd.DataFrame) -> dict:
    """Build a Vega radar chart spec from a correlation DataFrame.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Must have columns: covariate (str), abs_correlation (float 0-1)
    """
    table_values = [{"key": row["covariate"], "value": row["abs_correlation"]} for _, row in corr_df.iterrows()]

    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "description": "Covariate importance radar chart (Spearman |r|)",
        "width": 350,
        "height": 350,
        "padding": 60,
        "autosize": {"type": "none", "contains": "padding"},
        "title": {
            "text": "Covariate Importance",
            "subtitle": "Absolute Spearman correlation with disease cases",
            "anchor": "middle",
        },
        "signals": [{"name": "radius", "update": "width / 2"}],
        "data": [
            {"name": "table", "values": table_values},
            {
                "name": "keys",
                "source": "table",
                "transform": [{"type": "aggregate", "groupby": ["key"]}],
            },
        ],
        "scales": [
            {
                "name": "angular",
                "type": "point",
                "range": {"signal": "[-PI, PI]"},
                "padding": 0.5,
                "domain": {"data": "table", "field": "key"},
            },
            {
                "name": "radial",
                "type": "linear",
                "range": {"signal": "[0, radius]"},
                "zero": True,
                "nice": False,
                "domain": [0, 1],
            },
        ],
        "encode": {
            "enter": {
                "x": {"signal": "radius"},
                "y": {"signal": "radius"},
            }
        },
        "marks": [
            # Filled radar polygon
            {
                "type": "line",
                "name": "radar-line",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "interpolate": {"value": "linear-closed"},
                        "x": {"signal": "scale('radial', datum.value) * cos(scale('angular', datum.key))"},
                        "y": {"signal": "scale('radial', datum.value) * sin(scale('angular', datum.key))"},
                        "stroke": {"value": "steelblue"},
                        "strokeWidth": {"value": 2},
                        "fill": {"value": "steelblue"},
                        "fillOpacity": {"value": 0.15},
                    }
                },
            },
            # Data point dots
            {
                "type": "symbol",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "x": {"signal": "scale('radial', datum.value) * cos(scale('angular', datum.key))"},
                        "y": {"signal": "scale('radial', datum.value) * sin(scale('angular', datum.key))"},
                        "fill": {"value": "steelblue"},
                        "size": {"value": 40},
                        "tooltip": {
                            "signal": "{'Covariate': datum.key, 'Correlation |r|': format(datum.value, '.3f')}"
                        },
                    }
                },
            },
            # Value labels on each point
            {
                "type": "text",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "x": {"signal": "(scale('radial', datum.value) + 12) * cos(scale('angular', datum.key))"},
                        "y": {"signal": "(scale('radial', datum.value) + 12) * sin(scale('angular', datum.key))"},
                        "text": {"signal": "format(datum.value, '.2f')"},
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "fontSize": {"value": 10},
                        "fill": {"value": "#333"},
                    }
                },
            },
            # Radial grid lines (spokes)
            {
                "type": "rule",
                "name": "radial-grid",
                "from": {"data": "keys"},
                "zindex": 0,
                "encode": {
                    "enter": {
                        "x": {"value": 0},
                        "y": {"value": 0},
                        "x2": {"signal": "radius * cos(scale('angular', datum.key))"},
                        "y2": {"signal": "radius * sin(scale('angular', datum.key))"},
                        "stroke": {"value": "#ddd"},
                        "strokeWidth": {"value": 1},
                    }
                },
            },
            # Axis labels (covariate names)
            {
                "type": "text",
                "name": "key-label",
                "from": {"data": "keys"},
                "zindex": 1,
                "encode": {
                    "enter": {
                        "x": {"signal": "(radius + 14) * cos(scale('angular', datum.key))"},
                        "y": {"signal": "(radius + 14) * sin(scale('angular', datum.key))"},
                        "text": {"field": "key"},
                        "align": [
                            {
                                "test": "abs(scale('angular', datum.key)) > PI / 2",
                                "value": "right",
                            },
                            {"value": "left"},
                        ],
                        "baseline": [
                            {"test": "scale('angular', datum.key) > 0", "value": "top"},
                            {
                                "test": "scale('angular', datum.key) == 0",
                                "value": "middle",
                            },
                            {"value": "bottom"},
                        ],
                        "fill": {"value": "#333"},
                        "fontWeight": {"value": "bold"},
                        "fontSize": {"value": 11},
                    }
                },
            },
            # Outer boundary polygon
            {
                "type": "line",
                "name": "outer-line",
                "from": {"data": "radial-grid"},
                "encode": {
                    "enter": {
                        "interpolate": {"value": "linear-closed"},
                        "x": {"field": "x2"},
                        "y": {"field": "y2"},
                        "stroke": {"value": "#ddd"},
                        "strokeWidth": {"value": 1},
                    }
                },
            },
        ],
    }


@backtest_plot(
    plot_id="covariate_importance",
    name="Covariate Importance",
    description=(
        "Radar chart showing absolute Spearman correlation between each covariate "
        "and disease cases. Larger area = stronger overall covariate signal."
    ),
    needs_covariates=True,
)
class CovariateImportancePlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
        covariates: pd.DataFrame | None = None,
    ) -> PlotResult:
        if covariates is None or covariates.empty:
            return _empty_chart("No covariate data available")

        merged = observations.merge(covariates, on=["location", "time_period"], how="inner")
        covariate_cols = [c for c in merged.columns if c not in ("location", "time_period", "disease_cases")]

        if not covariate_cols:
            return _empty_chart("No covariates found in dataset")

        correlations = []
        for col in covariate_cols:
            valid = merged[["disease_cases", col]].dropna()
            if len(valid) > 3:
                result = stats.spearmanr(valid["disease_cases"], valid[col])
                corr_value = float(result.statistic)  # type: ignore[union-attr]
                correlations.append(
                    {
                        "covariate": col.replace("_", " ").title(),
                        "abs_correlation": abs(corr_value),
                    }
                )

        if not correlations:
            return _empty_chart("Insufficient data to compute correlations")

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

        return _build_radar_spec(corr_df)
