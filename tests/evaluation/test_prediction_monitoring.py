from chap_core.assessment.metrics.mae import MAEMetric
from chap_core.database.tables import MonitoringSource
from chap_core.services.prediction_monitoring_service import comparison_over_time


def test_comparison_over_time_gives_one_point_per_run(flat_observations, flat_forecasts):
    # A baseline that misses every observation by 3.
    baseline = flat_forecasts.assign(forecast=flat_observations.disease_cases.to_numpy() + 3)

    points = comparison_over_time(MAEMetric(), flat_observations, flat_forecasts, baseline, MonitoringSource.prediction)

    # Both forecast periods are horizons of a single run made after 2022W52.
    assert [point.period for point in points] == ["2022W52"]
    # MAE is 1.5 against the baseline's 3, so the skill ratio is 1 - 1.5 / 3.
    assert [(point.value, point.running_value, point.n_observed) for point in points] == [(0.5, 0.5, 4)]
    assert points[0].source == MonitoringSource.prediction


def test_comparison_over_time_scores_only_observed_forecasts_both_models_made(flat_observations, flat_forecasts):
    observations = flat_observations[flat_observations.time_period == "2023-W01"]
    baseline = flat_forecasts[flat_forecasts.location == "loc1"]

    points = comparison_over_time(MAEMetric(), observations, flat_forecasts, baseline, MonitoringSource.prediction)

    # Only loc1 in 2023-W01 is observed and forecast by both, with the same forecast.
    assert [(point.period, point.value, point.n_observed) for point in points] == [("2022W52", 0.0, 1)]


def test_comparison_over_time_without_baseline_forecasts_has_no_points(flat_observations, flat_forecasts):
    points = comparison_over_time(
        MAEMetric(), flat_observations, flat_forecasts, flat_forecasts.iloc[0:0], MonitoringSource.prediction
    )

    assert points == []
