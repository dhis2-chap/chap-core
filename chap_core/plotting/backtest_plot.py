import pandas as pd
from altair import FacetChart

from chap_core.assessment.flat_representations import convert_backtest_observations_to_flat_observations
from chap_core.database.tables import BackTest
import altair as alt

alt.data_transformers.enable("vegafusion")


def clean_time(period):
    """Convert period to ISO date format for Altair/vegafusion compatibility."""
    if len(period) == 6:
        # YYYYMM format -> YYYY-MM-01 (add day for full date)
        return f"{period[:4]}-{period[4:]}-01"
    elif len(period) == 7 and period[4] == '-':
        # YYYY-MM format -> YYYY-MM-01 (add day for full date)
        return f"{period}-01"
    else:
        return period

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
                    "time_period": clean_time(bt_forecast.period),
                    "location": bt_forecast.org_unit,
                    "split_period": clean_time(bt_forecast.last_seen_period),
                }
                | {f"q_{int(q * 100)}": v for q, v in zip(quantiles, bt_forecast.get_quantiles(quantiles))}
            )
        df = pd.DataFrame(rows)
        flat_observations = convert_backtest_observations_to_flat_observations(backtest.dataset.observations)
        flat_observations["time_period"] = flat_observations["time_period"].apply(clean_time)
        return cls(df, flat_observations)

    def plot(self) -> FacetChart:
        # Replicate observations for each split_period to show in all facets
        unique_split_periods = self._forecast["split_period"].unique()

        # Create observed data with all combinations of split_period
        observed_replicated = []
        for split_period in unique_split_periods:
            tmp = self._observed.copy()
            tmp["split_period"] = split_period
            observed_replicated.append(tmp)

        observed_with_split = pd.concat(observed_replicated, ignore_index=True )

        # Combine all data into a single dataset for faceting
        # Add a column to distinguish data types
        forecast_data = self._forecast.copy()
        forecast_data["data_type"] = "forecast"

        observed_data = observed_with_split.copy()
        observed_data["data_type"] = "observed"

        # Align column names - add disease_cases to forecast (as NaN) and quantiles to observed (as NaN)
        for col in ["q_10", "q_25", "q_50", "q_75", "q_90"]:
            if col not in observed_data.columns:
                observed_data[col] = None

        if "disease_cases" not in forecast_data.columns:
            forecast_data["disease_cases"] = None

        # Combine datasets
        combined_data = pd.concat([forecast_data, observed_data], ignore_index=True)

        # Create base chart with combined data
        base = alt.Chart(combined_data)

        # Forecast line (median)
        line = base.transform_filter(
            alt.datum.data_type == "forecast"
        ).mark_line().encode(
            x="time_period:T",
            y=alt.Y("q_50:Q", scale=alt.Scale(zero=False)),
        )

        # Error bands
        error1 = base.transform_filter(
            alt.datum.data_type == "forecast"
        ).mark_errorband(color="blue", opacity=0.3).encode(
            x="time_period:T",
            y=alt.Y("q_10:Q", scale=alt.Scale(zero=False)),
            y2="q_90:Q",
        )

        error2 = base.transform_filter(
            alt.datum.data_type == "forecast"
        ).mark_errorband(color="blue", opacity=0.5).encode(
            x="time_period:T",
            y=alt.Y("q_25:Q", scale=alt.Scale(zero=False)),
            y2="q_75:Q",
        )

        # Observations
        observations = base.transform_filter(
            alt.datum.data_type == "observed"
        ).mark_line(color="orange").encode(
            x="time_period:T",
            y=alt.Y("disease_cases:Q", scale=alt.Scale(zero=False)),
        )

        # Layer all components
        full_layer = error1 + error2 + line + observations

        # Facet the combined layer
        return full_layer.facet(
            column="split_period:O",
            row="location:N"
        ).resolve_scale(
            y="independent"
        ).properties(
            title="BackTest Forecasts with Observations"
        )
