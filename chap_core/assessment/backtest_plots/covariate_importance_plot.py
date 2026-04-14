"""
Covariate importance radar chart using Spearman correlation.

Shows a radar (spider) chart of signed Spearman rank correlations between
each covariate and the target variable (disease_cases). Each axis represents
a covariate. Positive correlations are shown in blue, negative in red.
The radial distance shows correlation magnitude (0-1), centered at zero.

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

    Each covariate is rendered as a spoke. The radial distance represents
    the absolute correlation magnitude. Positive correlations are blue,
    negative are red - matching the reference design.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Must have columns: covariate (str), correlation (float -1..1)
    """
    table_values = [
        {
            "key": row["covariate"],
            "value": abs(row["correlation"]),
            "signed": row["correlation"],
        }
        for _, row in corr_df.iterrows()
    ]

    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "description": "Covariate importance radar chart (signed Spearman r)",
        "width": 500,
        "height": 500,
        "padding": {"top": 80, "left": 70, "right": 70, "bottom": 70},
        "autosize": {"type": "none", "contains": "padding"},
        "title": {
            "text": "Covariate Importance",
            "subtitle": "Spearman correlation with disease cases",
            "anchor": "middle",
            "offset": 10,
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
            # Individual wedge bars per covariate (blue positive, red negative)
            {
                "type": "rule",
                "from": {"data": "table"},
                "zindex": 1,
                "encode": {
                    "enter": {
                        "x": {"value": 0},
                        "y": {"value": 0},
                        "x2": {"signal": "scale('radial', datum.value) * cos(scale('angular', datum.key))"},
                        "y2": {"signal": "scale('radial', datum.value) * sin(scale('angular', datum.key))"},
                        "stroke": {"signal": "datum.signed >= 0 ? 'steelblue' : 'indianred'"},
                        "strokeWidth": {"value": 8},
                        "strokeCap": {"value": "round"},
                        "tooltip": {"signal": "{'Covariate': datum.key, 'Spearman r': format(datum.signed, '+.3f')}"},
                    }
                },
            },
            # Data point dots at the end of each spoke
            {
                "type": "symbol",
                "from": {"data": "table"},
                "zindex": 2,
                "encode": {
                    "enter": {
                        "x": {"signal": "scale('radial', datum.value) * cos(scale('angular', datum.key))"},
                        "y": {"signal": "scale('radial', datum.value) * sin(scale('angular', datum.key))"},
                        "fill": {"signal": "datum.signed >= 0 ? 'steelblue' : 'indianred'"},
                        "size": {"value": 50},
                        "tooltip": {"signal": "{'Covariate': datum.key, 'Spearman r': format(datum.signed, '+.3f')}"},
                    }
                },
            },
            # Signed value labels
            {
                "type": "text",
                "from": {"data": "table"},
                "zindex": 2,
                "encode": {
                    "enter": {
                        "x": {"signal": "(scale('radial', datum.value) + 14) * cos(scale('angular', datum.key))"},
                        "y": {"signal": "(scale('radial', datum.value) + 14) * sin(scale('angular', datum.key))"},
                        "text": {"signal": "format(datum.signed, '+.3f')"},
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "fontSize": {"value": 10},
                        "fill": {"signal": "datum.signed >= 0 ? 'steelblue' : 'indianred'"},
                        "fontWeight": {"value": "bold"},
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
            # Legend: positive indicator
            {
                "type": "symbol",
                "encode": {
                    "enter": {
                        "x": {"signal": "radius + 30"},
                        "y": {"signal": "-radius + 10"},
                        "shape": {"value": "circle"},
                        "fill": {"value": "steelblue"},
                        "size": {"value": 80},
                    }
                },
            },
            {
                "type": "text",
                "encode": {
                    "enter": {
                        "x": {"signal": "radius + 40"},
                        "y": {"signal": "-radius + 10"},
                        "text": {"value": "Positive"},
                        "align": {"value": "left"},
                        "baseline": {"value": "middle"},
                        "fontSize": {"value": 11},
                        "fill": {"value": "#333"},
                    }
                },
            },
            # Legend: negative indicator
            {
                "type": "symbol",
                "encode": {
                    "enter": {
                        "x": {"signal": "radius + 30"},
                        "y": {"signal": "-radius + 28"},
                        "shape": {"value": "circle"},
                        "fill": {"value": "indianred"},
                        "size": {"value": 80},
                    }
                },
            },
            {
                "type": "text",
                "encode": {
                    "enter": {
                        "x": {"signal": "radius + 40"},
                        "y": {"signal": "-radius + 28"},
                        "text": {"value": "Negative"},
                        "align": {"value": "left"},
                        "baseline": {"value": "middle"},
                        "fontSize": {"value": 11},
                        "fill": {"value": "#333"},
                    }
                },
            },
        ],
    }


@backtest_plot(
    plot_id="covariate_importance",
    name="Covariate Importance",
    description=(
        "Radar chart showing Spearman correlation between each covariate "
        "and disease cases. Blue = positive, red = negative."
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
                # Skip constant covariates/targets where spearmanr returns NaN
                if pd.isna(corr_value):
                    continue
                correlations.append(
                    {
                        "covariate": col.replace("_", " ").title(),
                        "correlation": corr_value,
                    }
                )

        if not correlations:
            return _empty_chart("Insufficient data to compute correlations")

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)

        return _build_radar_spec(corr_df)
