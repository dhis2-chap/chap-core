"""
one numeric metric column per metric. get_metric validates this against your MetricSpec. If you try to return extra numeric columns (e.g., both value_diff and week_lag), CHAP will raise a “produced wrong columns” error.
"""
import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec

def _parse_year_week(week_str: str) -> tuple[int, int]:
    year_str, w = week_str.split("-W")
    return int(year_str), int(w)

def _week_index(week_str: str) -> int:
    y, w = _parse_year_week(week_str)
    return y * 54 + (w - 1)

def _week_diff(w1: str, w2: str) -> int:
    return _week_index(w2) - _week_index(w1)

def _pick_peak(rows: pd.DataFrame, value_col: str) -> tuple[str, float]:
    tmp = rows[['time_period', value_col]].copy()
    tmp['_wi'] = tmp['time_period'].map(_week_index)
    tmp = tmp.sort_values(by=[value_col, '_wi'], ascending=[False, True])
    top = tmp.iloc[0]
    return str(top['time_period']), float(top[value_col])

class PeakValueDiffMetric(MetricBase):
    spec = MetricSpec(output_dimensions=(DataDimension.location,), metric_name="Peak Value Difference")

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        fc_mean = (
            forecasts
            .groupby(['location', 'time_period'], as_index=False)['forecast']
            .mean()
            .rename(columns={'forecast': 'forecast_mean'})
        )

        obs = observations[['location', 'time_period', 'disease_cases']].copy()
        out_rows = []
        for loc, obs_loc in obs.groupby('location'):
            fc_loc = fc_mean[fc_mean['location'] == loc]
            if obs_loc.empty or fc_loc.empty:
                continue
            _, truth_val = _pick_peak(
                obs_loc[['time_period', 'disease_cases']],
                value_col='disease_cases'
            )
            _, pred_val = _pick_peak(
                fc_loc[['time_period', 'forecast_mean']],
                value_col='forecast_mean'
            )

            metric_val = float(truth_val - pred_val)
            out_rows.append({'location': loc, 'metric': metric_val})

        return pd.DataFrame(out_rows, columns=['location', 'metric'])


class PeakWeekLagMetric(MetricBase):
    spec = MetricSpec(
        output_dimensions=(DataDimension.location,),
        metric_name="Peak Week Lag",
        metric_id="peak_week_lag",
        description="Lag in weeks between true and predicted peak (pred - truth). Negative = model peaked earlier.",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        fc_mean = (
            forecasts
            .groupby(['location', 'time_period'], as_index=False)['forecast']
            .mean()
            .rename(columns={'forecast': 'forecast_mean'})
        )
        obs = observations[['location', 'time_period', 'disease_cases']].copy()

        out = []
        for loc, obs_loc in obs.groupby('location'):
            fc_loc = fc_mean[fc_mean['location'] == loc]
            if obs_loc.empty or fc_loc.empty:
                continue

            truth_timepoint, _ = _pick_peak(obs_loc, 'disease_cases')
            pred_timepoint, _ = _pick_peak(fc_loc, 'forecast_mean')

            lag_weeks = int(_week_diff(truth_timepoint, pred_timepoint))  # positive => predicted later
            out.append({'location': loc, 'metric': float(lag_weeks)})

        return pd.DataFrame(out, columns=['location', 'metric'])