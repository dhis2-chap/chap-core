from chap_core.database.tables import BackTestMetric
import pandas as pd

from chap_core.time_period import TimePeriod


def horizon_diff(period: str, period2: str) -> int:
    tp = TimePeriod.parse(period)
    tp2 = TimePeriod.parse(period2)
    return (tp - tp2) // tp.time_delta


def create_metric_table(metrics: list[BackTestMetric]):
    colnames = ["period", "org_unit", "value", "last_seen_period"]
    df = pd.DataFrame([metric.model_dump() for metric in metrics], columns=colnames)
    df["horizon"] = [horizon_diff(metric.period, metric.last_seen_period) for metric in metrics]
    return df
