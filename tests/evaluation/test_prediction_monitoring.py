from chap_core.assessment.metrics.mae import MAEMetric
from chap_core.database.tables import MonitoringSource
from chap_core.services.prediction_monitoring_service import metric_over_time


def test_metric_over_time_gives_one_point_per_run(flat_observations, flat_forecasts):
    points = metric_over_time(MAEMetric(), flat_observations, flat_forecasts, MonitoringSource.prediction)

    # Both forecast periods are horizons of a single run made after 2022W52.
    assert [point.period for point in points] == ["2022W52"]
    # Absolute errors are 1 and 2, pooled over locations and horizons.
    assert [(point.value, point.running_value, point.n_observed) for point in points] == [(1.5, 1.5, 4)]
    assert points[0].source == MonitoringSource.prediction


def test_metric_over_time_scores_only_observed_forecasts(flat_observations, flat_forecasts):
    observations = flat_observations[flat_observations.time_period == "2023-W01"]

    points = metric_over_time(MAEMetric(), observations, flat_forecasts, MonitoringSource.prediction)

    assert [(point.period, point.n_observed) for point in points] == [("2022W52", 2)]
