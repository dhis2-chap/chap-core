import math

import pytest

from chap_core.assessment.metrics import (
    CRPSMetric,
    Coverage10_90Metric,
    DataDimension,
    MAEMetric,
    MetricSpec,
    RatioAboveTruthMetric,
    SensitivityMetric,
    get_comparison_op,
    get_metrics_registry,
    list_comparison_ops,
    metric,
)


def test_skill_ratio_for_error_score():
    assert CRPSMetric().compare(0.8, 1.0) == pytest.approx(0.2)
    assert CRPSMetric().compare(1.5, 1.0) == pytest.approx(-0.5)


def test_skill_ratio_is_nan_when_reference_is_zero():
    assert math.isnan(CRPSMetric().compare(1.0, 0.0))


def test_difference_for_rate():
    assert SensitivityMetric().compare(0.7, 0.5) == pytest.approx(0.2)
    assert get_comparison_op("difference")(1.0, 3.0, MAEMetric.spec) == pytest.approx(2.0)


def test_target_distance_at_least_ignores_scores_above_target():
    coverage = Coverage10_90Metric()
    assert coverage.compare(0.9, 0.6) == pytest.approx(0.2)
    assert coverage.compare(0.7, 0.9) == pytest.approx(-0.1)
    assert coverage.compare(0.95, 0.85) == pytest.approx(0.0)


def test_target_distance_closest():
    assert RatioAboveTruthMetric().compare(0.6, 0.2) == pytest.approx(0.2)
    assert RatioAboveTruthMetric().compare(0.4, 0.6) == pytest.approx(0.0)


def test_metric_without_comparison_op_returns_none():
    class NoComparisonMAE(MAEMetric):
        spec = MetricSpec(metric_id="no_comparison_mae", metric_name="MAE")

    assert NoComparisonMAE().compare(1.0, 2.0) is None


def test_registering_unknown_comparison_op_fails():
    class BadMAE(MAEMetric):
        spec = MetricSpec(metric_id="bad_mae", metric_name="MAE", comparison_op="no_such_op")

    with pytest.raises(ValueError, match="no_such_op"):
        metric()(BadMAE)
    assert "bad_mae" not in get_metrics_registry()


@pytest.mark.parametrize("metric_id", sorted(get_metrics_registry()))
def test_every_registered_metric_has_a_working_comparison(metric_id):
    metric_cls = get_metrics_registry()[metric_id]
    assert metric_cls.spec.comparison_op in list_comparison_ops()
    assert math.isfinite(metric_cls().compare(0.4, 0.6))


def test_get_comparison_global(flat_observations, flat_forecasts_multiple_samples, flat_forecasts):
    crps = CRPSMetric()
    result = crps.get_comparison(flat_observations, flat_forecasts_multiple_samples, flat_forecasts)

    model = crps.get_global_metric(flat_observations, flat_forecasts_multiple_samples)["metric"].iloc[0]
    reference = crps.get_global_metric(flat_observations, flat_forecasts)["metric"].iloc[0]
    assert list(result.columns) == ["metric", "reference_metric", "comparison"]
    assert result["metric"].iloc[0] == pytest.approx(model)
    assert result["reference_metric"].iloc[0] == pytest.approx(reference)
    assert result["comparison"].iloc[0] == pytest.approx(1 - model / reference)


def test_get_comparison_per_location(flat_observations, flat_forecasts_multiple_samples, flat_forecasts):
    result = CRPSMetric().get_comparison(
        flat_observations, flat_forecasts_multiple_samples, flat_forecasts, dimensions=(DataDimension.location,)
    )
    assert list(result.columns) == ["location", "metric", "reference_metric", "comparison"]
    assert sorted(result["location"]) == ["loc1", "loc2"]
    assert (result["comparison"] > 0).all()


def test_get_comparison_uses_only_shared_cells(flat_observations, flat_forecasts_multiple_samples, flat_forecasts):
    reference = flat_forecasts[
        ~((flat_forecasts["location"] == "loc2") & (flat_forecasts["time_period"] == "2023-W02"))
    ]
    result = MAEMetric().get_comparison(flat_observations, flat_forecasts_multiple_samples, reference)

    # Absolute errors of the model medians are 1, 1, 2, 2; the loc2 2023-W02 cell (error 2) is dropped.
    assert result["metric"].iloc[0] == pytest.approx(4 / 3)
