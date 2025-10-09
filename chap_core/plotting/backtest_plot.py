import pandas as pd
from altair import FacetChart

from chap_core.assessment.flat_representations import convert_backtest_observations_to_flat_observations
from chap_core.database.tables import BackTest
import altair as alt


class BackTestPlot:
    def __init__(self, forecast_df: pd.DataFrame, observed_df: pd.DataFrame):
        self._forecast = forecast_df
        self._observed = observed_df

    @classmethod
    def from_backtest(cls, backtest: BackTest) -> "BackTestPlot":
        rows = []
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for bt_forecast in backtest.forecasts:
            rows.append(
                {
                    "time_period": bt_forecast.period,
                    "location": bt_forecast.org_unit,
                    "split_period": bt_forecast.last_seen_period,
                }
                | {f"q_{int(q * 100)}": v for q, v in zip(quantiles, bt_forecast.get_quantiles(quantiles))}
            )
        df = pd.DataFrame(rows)
        flat_observations = convert_backtest_observations_to_flat_observations(backtest.dataset.observations)
        return cls(df, flat_observations)

    def _error_band(self, q_low, q_high):
        error_band = (
            alt.Chart(self._forecast)
            .mark_errorband()
            .encode(
                x="time_period:T",
                y=f"q_{int(q_low * 100)}:Q",
                y2=f"q_{int(q_high * 100)}:Q",
            )
        )
        return error_band

    def plot(self) -> FacetChart:
        line = (
            alt.Chart(self._forecast)
            .mark_line()
            .encode(
                x="time_period:T",
                y="q_50:Q",
            )
        )

        error1 = self._error_band(0.1, 0.9)
        error2 = self._error_band(0.25, 0.75)
        observations = (
            alt.Chart(self._observed)
            .mark_line(color="black")
            .encode(
                x="time_period:T",
                y="disease_cases:Q",
            )
        )
        full_plot = line + error1 + error2 + observations
        return full_plot.facet(column="split_period:O", row="location:N").properties(
            title="BackTest Forecasts with Observations"
        )
