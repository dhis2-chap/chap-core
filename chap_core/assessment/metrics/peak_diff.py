import re
import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec

_WEEK_RE = re.compile(r"^(\d{4})-W(\d{2})$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")


def _time_index(tp: str) -> int:
    string = str(tp)

    index = _WEEK_RE.match(string)

    if index:
        year = int(index.group(1))
        week = int(index.group(2))
        return year * 54 + (week - 1)

    index = _MONTH_RE.match(string)
    if index:
        year = int(index.group(1))
        month = int(index.group(2))
        return year * 12 + (month - 1)

    dt = pd.to_datetime(string)
    return int(dt.to_period("D").ordinal)


def _time_diff(tp1: str, tp2: str) -> int:
    return _time_index(tp2) - _time_index(tp1)


def _pick_peak(rows: pd.DataFrame, value_col: str) -> tuple[str, float]:
    tmp = rows[["time_period", value_col]].copy()
    tmp["_ti"] = tmp["time_period"].map(_time_index)
    tmp = tmp.sort_values(by=[value_col, "_ti"], ascending=[False, True])
    top = tmp.iloc[0]
    return str(top["time_period"]), float(top[value_col])


class PeakValueDiffMetric(MetricBase):
    # Now returns (location, time_period, horizon_distance, metric)
    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="Peak Value Difference",
        metric_id="peak_value_diff",
        description="Truth peak value minus predicted peak value, per horizon.",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        fc_mean = (
            forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)["forecast"]
            .mean()
            .rename(columns={"forecast": "forecast_mean"})
        )

        obs = observations[["location", "time_period", "disease_cases"]].copy()

        out_rows = []
        for loc, obs_loc in obs.groupby("location"):
            truth_timepoint, truth_val = _pick_peak(obs_loc, "disease_cases")

            fc_loc = fc_mean[fc_mean["location"] == loc]
            if fc_loc.empty:
                continue

            for h, fc_loc_h in fc_loc.groupby("horizon_distance"):
                if fc_loc_h.empty:
                    continue
                _, pred_val = _pick_peak(fc_loc_h[["time_period", "forecast_mean"]], "forecast_mean")

                metric_val = float(truth_val - pred_val)
                out_rows.append(
                    {
                        "location": loc,
                        "time_period": truth_timepoint,
                        "horizon_distance": int(h),
                        "metric": metric_val,
                    }
                )

        return pd.DataFrame(out_rows, columns=["location", "time_period", "horizon_distance", "metric"])


class PeakPeriodLagMetric(MetricBase):
    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="Peak Period Lag",
        metric_id="peak_period_lag",
        description="Lag in time periods (weeks for weekly data, months for monthly data) between true and predicted peak (pred - truth), per horizon.",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        fc_mean = (
            forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)["forecast"]
            .mean()
            .rename(columns={"forecast": "forecast_mean"})
        )
        obs = observations[["location", "time_period", "disease_cases"]].copy()

        out_rows = []
        for loc, obs_loc in obs.groupby("location"):
            truth_timepoint, _ = _pick_peak(obs_loc, "disease_cases")

            fc_loc = fc_mean[fc_mean["location"] == loc]
            if fc_loc.empty:
                continue

            for h, fc_loc_h in fc_loc.groupby("horizon_distance"):
                if fc_loc_h.empty:
                    continue
                pred_timepoint, _ = _pick_peak(fc_loc_h[["time_period", "forecast_mean"]], "forecast_mean")

                lag = int(_time_diff(truth_timepoint, pred_timepoint))
                out_rows.append(
                    {
                        "location": loc,
                        "time_period": truth_timepoint,
                        "horizon_distance": int(h),
                        "metric": float(lag),
                    }
                )

        return pd.DataFrame(out_rows, columns=["location", "time_period", "horizon_distance", "metric"])
