from abc import ABC, abstractmethod

import pandas as pd
from altair import FacetChart

from chap_core.assessment.evaluation import Evaluation
from chap_core.database.tables import BackTest
import altair as alt
import textwrap

alt.data_transformers.enable("vegafusion")


def title_chart(text: str, width: int = 600, font_size: int = 24, pad: int = 10):
    """Return an Altair chart that just displays a title."""
    return (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_text(
            text=text,
            fontSize=font_size,
            fontWeight="bold",
            align="center",
            baseline="top",
        )
        .properties(width=width, height=font_size + pad)
    )


def text_chart(text, line_length=80, font_size=12, align="left", pad_bottom=50):
    import altair as alt
    import pandas as pd

    lines = textwrap.wrap(text, width=line_length)
    df = pd.DataFrame({"line": lines, "y": range(len(lines))})

    line_spacing = font_size + 2
    total_height = len(lines) * line_spacing + pad_bottom

    chart = (
        alt.Chart(df)
        .mark_text(align=align, baseline="top", fontSize=font_size)
        .encode(text="line", y=alt.Y("y:O", axis=None))
        .properties(height=total_height)
    )
    return chart


def clean_time(period):
    """Convert period to ISO date format for Altair/vegafusion compatibility."""
    if len(period) == 6:
        # YYYYMM format -> YYYY-MM-01 (add day for full date)
        return f"{period[:4]}-{period[4:]}-01"
    elif len(period) == 7 and period[4] == "-":
        # YYYY-MM format -> YYYY-MM-01 (add day for full date)
        return f"{period}-01"
    else:
        return period


class BackTestPlotBase(ABC):
    """
    Abstract base class for backtest plotting.

    Subclasses must implement:
    - from_backtest: Class method to create plot instance from a BackTest object
    - plot: Method to generate and return the visualization
    - name: Class variable with the name of the plot type
    """

    name: str = ""

    @classmethod
    @abstractmethod
    def from_backtest(cls, backtest: BackTest):
        """
        Create a plot instance from a BackTest object.

        Parameters
        ----------
        backtest : BackTest
            The backtest object containing forecast and observation data

        Returns
        -------
        BackTestPlotBase
            An instance of the concrete plot class
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Generate and return the visualization.

        Returns
        -------
        Chart object (implementation-specific)
            The visualization object (e.g., FacetChart for Altair-based plots)
        """
        pass


class EvaluationBackTestPlot(BackTestPlotBase):
    """
    Backtest-plot that shows truth vs predictions over time.

    Shows forecasts with uncertainty bands and observed values.
    Optionally includes historical observations for context.
    """

    name: str = "Evaluation Plot"

    def __init__(
        self,
        forecast_df: pd.DataFrame,
        observed_df: pd.DataFrame,
        historical_df: pd.DataFrame = None,
    ):
        self._forecast = forecast_df
        self._observed = observed_df
        self._historical = historical_df

    @classmethod
    def from_evaluation(cls, evaluation: Evaluation) -> "EvaluationBackTestPlot":
        """
        Create plot from an Evaluation object.

        This method preserves historical observations if present in the Evaluation.

        Parameters
        ----------
        evaluation : Evaluation
            The evaluation object containing forecasts, observations, and optional
            historical observations

        Returns
        -------
        EvaluationBackTestPlot
            An instance ready for plotting
        """
        backtest = evaluation.to_backtest()

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

        flat_data = evaluation.to_flat()

        # Test period observations
        flat_observations = pd.DataFrame(flat_data.observations).copy()
        flat_observations["time_period"] = flat_observations["time_period"].apply(clean_time)

        # Historical observations (optional)
        historical_df = None
        if flat_data.historical_observations is not None:
            historical_df = pd.DataFrame(flat_data.historical_observations).copy()
            historical_df["time_period"] = historical_df["time_period"].apply(clean_time)

        return cls(df, flat_observations, historical_df)

    @classmethod
    def from_backtest(cls, backtest: BackTest) -> "EvaluationBackTestPlot":
        """
        Create plot from a BackTest object.

        Note: This method does not include historical observations.
        Use from_evaluation() to include historical context.

        Parameters
        ----------
        backtest : BackTest
            The backtest object containing forecast and observation data

        Returns
        -------
        EvaluationBackTestPlot
            An instance ready for plotting
        """
        evaluation = Evaluation.from_backtest(backtest)
        return cls.from_evaluation(evaluation)

    def plot(self) -> FacetChart:
        # Replicate observations for each split_period to show in all facets
        unique_split_periods = self._forecast["split_period"].unique()

        # Create observed data (test periods) with all combinations of split_period
        observed_replicated = []
        for split_period in unique_split_periods:
            tmp = self._observed.copy()
            tmp["split_period"] = split_period
            observed_replicated.append(tmp)

        observed_with_split = pd.concat(observed_replicated, ignore_index=True)

        # Add historical observations if available
        # Historical observations are filtered to only show up to each split_period
        if self._historical is not None and not self._historical.empty:
            historical_replicated = []
            for split_period in unique_split_periods:
                tmp = self._historical.copy()
                tmp["split_period"] = split_period
                # Filter to only include observations before or at the split_period
                tmp = tmp[tmp["time_period"] <= split_period]
                historical_replicated.append(tmp)

            historical_with_split = pd.concat(historical_replicated, ignore_index=True)

            # Combine historical and test observations
            # Remove duplicates (historical may overlap with test observations)
            all_observations = pd.concat(
                [historical_with_split, observed_with_split], ignore_index=True
            ).drop_duplicates(subset=["location", "time_period", "split_period"])
        else:
            all_observations = observed_with_split

        # Combine all data into a single dataset for faceting
        # Add a column to distinguish data types
        forecast_data = self._forecast.copy()
        forecast_data["data_type"] = "forecast"

        observed_data = all_observations.copy()
        observed_data["data_type"] = "observed"

        # Align column names - add disease_cases to forecast (as NaN) and quantiles to observed (as NaN)
        for col in ["q_10", "q_25", "q_50", "q_75", "q_90"]:
            if col not in observed_data.columns:
                observed_data[col] = None

        if "disease_cases" not in forecast_data.columns:
            forecast_data["disease_cases"] = None

        # Drop all-NA columns before concatenation to avoid FutureWarning
        forecast_data = forecast_data.dropna(axis=1, how="all")
        observed_data = observed_data.dropna(axis=1, how="all")

        # Combine datasets
        combined_data = pd.concat([forecast_data, observed_data], ignore_index=True)

        # Create base chart with combined data
        base = alt.Chart(combined_data)

        # Forecast line (median)
        line = (
            base.transform_filter(alt.datum.data_type == "forecast")
            .mark_line()
            .encode(
                x="time_period:T",
                y=alt.Y("q_50:Q", scale=alt.Scale(zero=False)),
            )
        )

        # Error bands
        error1 = (
            base.transform_filter(alt.datum.data_type == "forecast")
            .mark_errorband(color="blue", opacity=0.3)
            .encode(
                x="time_period:T",
                y=alt.Y("q_10:Q", scale=alt.Scale(zero=False)),
                y2="q_90:Q",
            )
        )

        error2 = (
            base.transform_filter(alt.datum.data_type == "forecast")
            .mark_errorband(color="blue", opacity=0.5)
            .encode(
                x="time_period:T",
                y=alt.Y("q_25:Q", scale=alt.Scale(zero=False)),
                y2="q_75:Q",
            )
        )

        # Observations (including historical context if available)
        observations = (
            base.transform_filter(alt.datum.data_type == "observed")
            .mark_line(color="orange")
            .encode(
                x="time_period:T",
                y=alt.Y("disease_cases:Q", scale=alt.Scale(zero=False)),
            )
        )

        # Layer all components
        full_layer = error1 + error2 + line + observations

        # Facet the combined layer
        return (
            full_layer.facet(column="split_period:O", row="location:N")
            .resolve_scale(y="independent")
            .properties(title="BackTest Forecasts with Observations")
        )
