"""
Covariate importance column chart.

One bar per covariate, height equal to the signed Spearman rank correlation
between that covariate and disease_cases over the joint observation window.
Bars are sorted by signed correlation descending — strongest positive on the
left, strongest negative on the right — and colored steelblue when the
correlation is positive, red when negative. The y-axis is fixed to [-1, 1]
so visual length is comparable across runs and a zero rule marks the no-effect
line.
"""

import altair as alt
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot

POSITIVE_COLOR = "#4682b4"
NEGATIVE_COLOR = "#cd5c5c"
MIN_VALID_PAIRS = 4


def _message_chart(message: str) -> ChartType:
    """Render a centered text message inside an empty chart frame."""
    return (  # type: ignore[no-any-return]
        alt.Chart(pd.DataFrame({"text": [message]}))
        .mark_text(fontSize=14, color="#666")
        .encode(text="text:N")
        .properties(width="container", height=120, title="Covariate Importance")
    )


@backtest_plot(
    plot_id="covariate_importance",
    name="Covariate Importance",
    description=(
        "Signed Spearman correlation between each covariate and disease cases, "
        "shown as a column chart. Steelblue = positive, red = negative."
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
            return _message_chart("No covariate data available")

        merged = observations.merge(covariates, on=["location", "time_period"], how="inner")
        covariate_cols = [c for c in merged.columns if c not in ("location", "time_period", "disease_cases")]
        if not covariate_cols:
            return _message_chart("No covariates found in dataset")

        rows: list[dict] = []
        for col in covariate_cols:
            valid = merged[["disease_cases", col]].dropna()
            if len(valid) < MIN_VALID_PAIRS:
                continue
            result = stats.spearmanr(valid["disease_cases"], valid[col])
            corr_value = float(result.statistic)  # type: ignore[union-attr]
            if pd.isna(corr_value):
                continue
            rows.append({"covariate": col.replace("_", " ").title(), "correlation": corr_value})

        if not rows:
            return _message_chart("Insufficient data to compute correlations")

        corr_df = pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)
        sort_order = corr_df["covariate"].tolist()

        bars = (
            alt.Chart(corr_df)
            .mark_bar()
            .encode(
                x=alt.X("covariate:N", sort=sort_order, title="Covariate", axis=alt.Axis(labelAngle=-30)),
                y=alt.Y(
                    "correlation:Q",
                    title="Spearman correlation",
                    scale=alt.Scale(domain=[-1, 1]),
                ),
                color=alt.condition(
                    "datum.correlation < 0",
                    alt.value(NEGATIVE_COLOR),
                    alt.value(POSITIVE_COLOR),
                ),
                tooltip=[
                    alt.Tooltip("covariate:N", title="Covariate"),
                    alt.Tooltip("correlation:Q", format="+.3f", title="Spearman r"),
                ],
            )
        )

        zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888", strokeWidth=1).encode(y="y:Q")

        return (  # type: ignore[no-any-return]
            (bars + zero_rule)
            .properties(width="container", height=500, title="Covariate Importance")
            .configure_view(strokeWidth=0)
        )
