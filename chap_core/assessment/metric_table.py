import pandas as pd

from chap_core.database.tables import BackTestMetric
from chap_core.time_period import TimePeriod


def horizon_diff(period: str, period2: str) -> int:
    """Calculate the 1-based horizon distance between two time periods."""
    tp = TimePeriod.parse(period)
    tp2 = TimePeriod.parse(period2)
    return int((tp - tp2) // tp.time_delta) + 1


def create_metric_table(metrics: list[BackTestMetric]):
    colnames = ["period", "org_unit", "value", "last_seen_period"]
    df = pd.DataFrame([metric.model_dump() for metric in metrics], columns=colnames)
    df["horizon"] = [horizon_diff(metric.period, metric.last_seen_period) for metric in metrics]
    return df
