"""
Covariate importance chart using Spearman correlation.

Shows a diverging horizontal bar chart of Spearman rank correlations between
each covariate and the target variable (disease_cases). Positive correlations
extend right (blue), negative extend left (red).

Inspired by the Uganda Nutrition Early Warning tool's covariate importance
chart (CLIM-538).
"""

from typing import cast

import altair as alt
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot


def _empty_chart(message: str) -> ChartType:
    return cast(
        "ChartType",
        alt.Chart(pd.DataFrame({"text": [message]}))
        .mark_text(fontSize=14)
        .encode(text="text:N")
        .properties(width=400, height=100),
    )


@backtest_plot(
    plot_id="covariate_importance",
    name="Covariate Importance",
    description=(
        "Diverging bar chart showing Spearman correlation between each covariate "
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
    ) -> ChartType:
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
                        "correlation": corr_value,
                        "abs_correlation": abs(corr_value),
                        "direction": "Positive" if corr_value >= 0 else "Negative",
                    }
                )

        if not correlations:
            return _empty_chart("Insufficient data to compute correlations")

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values("abs_correlation", ascending=True).reset_index(drop=True)

        bars = (
            alt.Chart(corr_df)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X(
                    "correlation:Q",
                    title="Spearman Correlation",
                    scale=alt.Scale(domain=[-1, 1]),
                    axis=alt.Axis(format=".2f"),
                ),
                y=alt.Y("covariate:N", title=None, sort="-x"),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["Positive", "Negative"], range=["steelblue", "indianred"]),
                    title="Direction",
                ),
                tooltip=[
                    alt.Tooltip("covariate:N", title="Covariate"),
                    alt.Tooltip("correlation:Q", format="+.3f", title="Spearman r"),
                ],
            )
        )

        # Zero reference line
        rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray", strokeDash=[2, 2]).encode(x="x:Q")

        chart = alt.layer(bars, rule).properties(
            width=400, height=max(len(corr_df) * 35, 120), title="Covariate Importance (Spearman correlation)"
        )

        return chart  # type: ignore[no-any-return]
