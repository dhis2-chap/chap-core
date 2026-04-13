"""
Covariate importance radial bar chart using Spearman correlation.

Shows a radial bar chart of Spearman rank correlations between each covariate
and the target variable (disease_cases). Positive correlations are shown in blue,
negative in red.

Inspired by the Uganda Nutrition Early Warning tool's covariate importance
radar chart (CLIM-538).
"""

from typing import cast

import altair as alt
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot


@backtest_plot(
    plot_id="covariate_importance",
    name="Covariate Importance",
    description=(
        "Radial bar chart showing Spearman correlation between each covariate "
        "and disease cases. Blue = positive correlation, red = negative."
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
    ) -> ChartType:
        if covariates is None or covariates.empty:
            return cast(
                "ChartType",
                (
                    alt.Chart(pd.DataFrame({"text": ["No covariate data available"]}))
                    .mark_text(fontSize=16)
                    .encode(text="text:N")
                    .properties(width=400, height=200)
                ),
            )

        # Merge observations with covariates to compute correlations
        merged = observations.merge(covariates, on=["location", "time_period"], how="inner")

        # Identify covariate columns (everything except location, time_period, disease_cases)
        covariate_cols = [c for c in merged.columns if c not in ("location", "time_period", "disease_cases")]

        if not covariate_cols:
            return cast(
                "ChartType",
                (
                    alt.Chart(pd.DataFrame({"text": ["No covariates found in dataset"]}))
                    .mark_text(fontSize=16)
                    .encode(text="text:N")
                    .properties(width=400, height=200)
                ),
            )

        # Compute Spearman correlation for each covariate vs disease_cases
        correlations = []
        for col in covariate_cols:
            valid = merged[["disease_cases", col]].dropna()
            if len(valid) > 3:
                result = stats.spearmanr(valid["disease_cases"], valid[col])
                corr_value = float(result.statistic)  # type: ignore[union-attr]
                correlations.append(
                    {
                        "covariate": col.replace("_", " ").title(),
                        "correlation": corr_value,
                        "abs_correlation": abs(corr_value),
                        "direction": "positive" if corr_value >= 0 else "negative",
                        "label": f"{corr_value:+.3f}",
                    }
                )

        if not correlations:
            return cast(
                "ChartType",
                (
                    alt.Chart(pd.DataFrame({"text": ["Insufficient data to compute correlations"]}))
                    .mark_text(fontSize=16)
                    .encode(text="text:N")
                    .properties(width=400, height=200)
                ),
            )

        corr_df = pd.DataFrame(correlations)

        # Sort by absolute correlation for visual clarity
        corr_df = corr_df.sort_values("abs_correlation", ascending=True).reset_index(drop=True)

        # Radial bar chart using Altair
        bars = (
            alt.Chart(corr_df)
            .mark_arc(innerRadius=30, stroke="white", strokeWidth=1)
            .encode(
                theta=alt.Theta("abs_correlation:Q", stack=True, scale=alt.Scale(domain=[0, len(corr_df)])),
                radius=alt.Radius(
                    "abs_correlation:Q", scale=alt.Scale(type="sqrt", zero=True, rangeMin=30, rangeMax=180)
                ),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["positive", "negative"], range=["steelblue", "indianred"]),
                    title="Correlation",
                ),
                tooltip=[
                    alt.Tooltip("covariate:N", title="Covariate"),
                    alt.Tooltip("label:N", title="Spearman r"),
                ],
                order=alt.Order("abs_correlation:Q", sort="ascending"),
            )
        )

        # Add labels
        labels = (
            alt.Chart(corr_df)
            .mark_text(radiusOffset=15, fontSize=10)
            .encode(
                theta=alt.Theta("abs_correlation:Q", stack=True, scale=alt.Scale(domain=[0, len(corr_df)])),
                radius=alt.Radius(
                    "abs_correlation:Q", scale=alt.Scale(type="sqrt", zero=True, rangeMin=30, rangeMax=180)
                ),
                text="covariate:N",
                order=alt.Order("abs_correlation:Q", sort="ascending"),
            )
        )

        chart = (
            alt.layer(bars, labels)
            .properties(width=400, height=400, title="Covariate Importance (Spearman correlation with disease cases)")
            .resolve_scale(theta="independent")
        )

        return chart  # type: ignore[no-any-return]
