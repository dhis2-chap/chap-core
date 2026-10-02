from chap_core.assessment.metrics.mae import MAEMetric
from chap_core.database.tables import MonitoringSource
from chap_core.services.prediction_monitoring_service import metric_over_time


def test_metric_over_time_scores_each_period_and_running_aggregate(flat_observations, flat_forecasts):
    points = metric_over_time(MAEMetric(), flat_observations, flat_forecasts, MonitoringSource.prediction)

    assert [point.period for point in points] == ["2023-W01", "2023-W02"]
    # Absolute errors are 1 and 2 in both periods, pooled over locations.
    assert [point.value for point in points] == [1.5, 1.5]
    assert [point.running_value for point in points] == [1.5, 1.5]
    assert {point.source for point in points} == {MonitoringSource.prediction}


def test_metric_over_time_skips_periods_without_observations(flat_observations, flat_forecasts):
    observations = flat_observations[flat_observations.time_period == "2023-W01"]

    points = metric_over_time(MAEMetric(), observations, flat_forecasts, MonitoringSource.prediction)

    assert [point.period for point in points] == ["2023-W01"]
