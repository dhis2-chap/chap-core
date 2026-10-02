import numpy as np
import pytest

from chap_core.assessment.flat_representations import DataDimension
from chap_core.assessment.metrics.outbreak_classification import (
    FalseAlarmRateMetric,
    MatthewsCorrelationMetric,
    OutbreakF1Metric,
    OutbreakPrecisionMetric,
)
from chap_core.assessment.metrics.outbreak_probability import (
    LOG_SCORE_CLIP,
    BrierScoreMetric,
    BrierSkillScoreMetric,
    OutbreakLogScoreMetric,
)

_GLOBAL_ONLY = (OutbreakF1Metric, MatthewsCorrelationMetric, BrierSkillScoreMetric)


def _global(metric_cls, scenario):
    historical, observations, forecasts = scenario
    metric = metric_cls(historical_observations=historical)
    return metric.get_global_metric(observations, forecasts).iloc[0]["metric"]


def test_precision_counts_only_the_alerts(balanced_alert_scenario):
    """One of the two raised alerts was a real outbreak."""
    assert _global(OutbreakPrecisionMetric, balanced_alert_scenario) == pytest.approx(0.5)


def test_false_alarm_rate_counts_only_the_quiet_periods(balanced_alert_scenario):
    """One of the two quiet periods was alerted anyway."""
    assert _global(FalseAlarmRateMetric, balanced_alert_scenario) == pytest.approx(0.5)


def test_f1_is_the_harmonic_mean_of_precision_and_sensitivity(balanced_alert_scenario):
    """Precision and sensitivity are both 0.5 here, so F1 is too."""
    assert _global(OutbreakF1Metric, balanced_alert_scenario) == pytest.approx(0.5)


def test_matthews_correlation_is_zero_for_chance_agreement(balanced_alert_scenario):
    """A balanced confusion matrix means alerts carry no information at all."""
    assert _global(MatthewsCorrelationMetric, balanced_alert_scenario) == pytest.approx(0.0)


def test_brier_score_is_the_mean_squared_error(balanced_alert_scenario):
    """Probabilities are 1, 0, 1, 0 against outcomes 1, 1, 0, 0 -- two cells wrong outright."""
    assert _global(BrierScoreMetric, balanced_alert_scenario) == pytest.approx(0.5)


def test_brier_skill_is_negative_when_worse_than_the_base_rate(balanced_alert_scenario):
    """Half the cells are outbreaks, so climatology scores 0.25 and the model 0.5."""
    assert _global(BrierSkillScoreMetric, balanced_alert_scenario) == pytest.approx(-1.0)


def test_log_score_clips_confident_misses(balanced_alert_scenario):
    """Two cells are confidently wrong; the clip makes each cost -log(eps) rather than infinity."""
    expected = (2 * -np.log(LOG_SCORE_CLIP) + 2 * -np.log(1 - LOG_SCORE_CLIP)) / 4
    assert _global(OutbreakLogScoreMetric, balanced_alert_scenario) == pytest.approx(expected)
    assert np.isfinite(expected)


@pytest.mark.parametrize("metric_cls", _GLOBAL_ONLY, ids=lambda c: c.spec.metric_id)
def test_global_only_metrics_refuse_a_breakdown(metric_cls, balanced_alert_scenario):
    """A ratio of aggregates has no per-cell value, so asking for one is an error, not a wrong number."""
    historical, observations, forecasts = balanced_alert_scenario
    metric = metric_cls(historical_observations=historical)
    with pytest.raises(ValueError, match="cannot be broken down"):
        metric.get_metric(observations, forecasts, dimensions=(DataDimension.location,))


@pytest.mark.parametrize("metric_cls", _GLOBAL_ONLY, ids=lambda c: c.spec.metric_id)
def test_global_only_metrics_declare_no_output_dimensions(metric_cls):
    """This is the contract compute_all_detailed_metrics keys on to skip them."""
    assert metric_cls.spec.output_dimensions == ()


def test_brier_skill_is_undefined_without_both_outcomes(two_season_history, make_flat_forecasts):
    """With no quiet period the climatological reference is perfect and skill has no meaning."""
    import pandas as pd

    observations = pd.DataFrame([{"location": "A", "time_period": "2023-06", "disease_cases": 200.0}])
    forecasts = make_flat_forecasts("A", "2023-06", 1, [300.0] * 10)
    metric = BrierSkillScoreMetric(historical_observations=two_season_history)
    assert np.isnan(metric.get_global_metric(observations, forecasts).iloc[0]["metric"])


def test_only_two_sided_metrics_are_optimization_objectives():
    """Precision and false-alarm rate are each maximised by a degenerate model, like sensitivity before them."""
    assert OutbreakPrecisionMetric.spec.optimization_direction is None
    assert FalseAlarmRateMetric.spec.optimization_direction is None
    for metric_cls in _GLOBAL_ONLY + (BrierScoreMetric, OutbreakLogScoreMetric):
        assert metric_cls.spec.optimization_direction is not None


def test_plot_metric_picker_omits_unplottable_metrics():
    """Every metric plot breaks a score down by a dimension a global-only metric has no values for."""
    from chap_core.rest_api.v1.routers.visualization import get_available_metrics

    offered = {metric.id for metric in get_available_metrics(backtest_id=1)}
    assert "outbreak_brier" in offered
    for metric_cls in _GLOBAL_ONLY:
        assert metric_cls.spec.metric_id not in offered
