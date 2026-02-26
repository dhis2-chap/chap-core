import numpy as np
import pandas as pd
import pytest

from chap_core.assessment.metrics.outbreak_detection import (
    SensitivityMetric,
    SpecificityMetric,
    compute_seasonal_thresholds,
)


@pytest.fixture()
def historical_observations():
    """Historical data where location A in month 6 has mean=100, std=10 -> threshold=120."""
    rows = []
    for year in range(2018, 2023):
        rows.append({"location": "A", "time_period": f"{year}-06-15", "disease_cases": 100.0})
    # std of [100,100,100,100,100] = 0, so add variance
    # Use values: 90, 95, 100, 105, 110 -> mean=100, std=~7.07 -> threshold=~114.14
    rows = []
    values = [90.0, 95.0, 100.0, 105.0, 110.0]
    for year, val in zip(range(2018, 2023), values):
        rows.append({"location": "A", "time_period": f"{year}-06-15", "disease_cases": val})
    return pd.DataFrame(rows)


@pytest.fixture()
def threshold_value():
    """Expected threshold for historical_observations fixture."""
    values = [90.0, 95.0, 100.0, 105.0, 110.0]
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    return mean + 2 * std


def test_compute_seasonal_thresholds(historical_observations, threshold_value):
    result = compute_seasonal_thresholds(historical_observations)
    assert list(result.columns) == ["location", "month", "threshold"]
    assert len(result) == 1
    assert result.iloc[0]["location"] == "A"
    assert result.iloc[0]["month"] == 6
    assert result.iloc[0]["threshold"] == pytest.approx(threshold_value)


def _make_forecasts(location, time_period, horizon, samples):
    """Helper to create a forecast DataFrame from sample values."""
    return pd.DataFrame(
        {
            "location": location,
            "time_period": time_period,
            "horizon_distance": horizon,
            "sample": list(range(len(samples))),
            "forecast": samples,
        }
    )


def test_sensitivity_true_positive(historical_observations, threshold_value):
    """When there is an outbreak and the forecast triggers an alert -> metric=1.0."""
    observations = pd.DataFrame([{"location": "A", "time_period": "2023-06-15", "disease_cases": threshold_value + 10}])
    # All samples above threshold -> alert
    forecasts = _make_forecasts("A", "2023-06-15", 1, [threshold_value + 5] * 10)
    metric = SensitivityMetric(historical_observations=historical_observations)
    result = metric.get_detailed_metric(observations, forecasts)
    assert len(result) == 1
    assert result.iloc[0]["metric"] == 1.0


def test_sensitivity_false_negative(historical_observations, threshold_value):
    """When there is an outbreak but forecast does not alert -> metric=0.0."""
    observations = pd.DataFrame([{"location": "A", "time_period": "2023-06-15", "disease_cases": threshold_value + 10}])
    # All samples below threshold -> no alert
    forecasts = _make_forecasts("A", "2023-06-15", 1, [threshold_value - 5] * 10)
    metric = SensitivityMetric(historical_observations=historical_observations)
    result = metric.get_detailed_metric(observations, forecasts)
    assert len(result) == 1
    assert result.iloc[0]["metric"] == 0.0


def test_specificity_true_negative(historical_observations, threshold_value):
    """When there is no outbreak and no alert -> metric=1.0."""
    observations = pd.DataFrame([{"location": "A", "time_period": "2023-06-15", "disease_cases": threshold_value - 10}])
    # All samples below threshold -> no alert
    forecasts = _make_forecasts("A", "2023-06-15", 1, [threshold_value - 5] * 10)
    metric = SpecificityMetric(historical_observations=historical_observations)
    result = metric.get_detailed_metric(observations, forecasts)
    assert len(result) == 1
    assert result.iloc[0]["metric"] == 1.0


def test_specificity_false_positive(historical_observations, threshold_value):
    """When there is no outbreak but forecast alerts -> metric=0.0."""
    observations = pd.DataFrame([{"location": "A", "time_period": "2023-06-15", "disease_cases": threshold_value - 10}])
    # All samples above threshold -> alert (false positive)
    forecasts = _make_forecasts("A", "2023-06-15", 1, [threshold_value + 5] * 10)
    metric = SpecificityMetric(historical_observations=historical_observations)
    result = metric.get_detailed_metric(observations, forecasts)
    assert len(result) == 1
    assert result.iloc[0]["metric"] == 0.0


def test_sensitivity_global_aggregation(historical_observations, threshold_value):
    """Global sensitivity with mixed TP and FN gives correct rate."""
    observations = pd.DataFrame(
        [
            {"location": "A", "time_period": "2023-06-15", "disease_cases": threshold_value + 10},
            {"location": "A", "time_period": "2023-06-22", "disease_cases": threshold_value + 10},
        ]
    )
    # First: alert (TP), Second: no alert (FN)
    forecasts = pd.concat(
        [
            _make_forecasts("A", "2023-06-15", 1, [threshold_value + 5] * 10),
            _make_forecasts("A", "2023-06-22", 1, [threshold_value - 5] * 10),
        ],
        ignore_index=True,
    )
    metric = SensitivityMetric(historical_observations=historical_observations)
    result = metric.get_global_metric(observations, forecasts)
    assert result.iloc[0]["metric"] == pytest.approx(0.5)


def test_sensitivity_no_outbreaks(historical_observations, threshold_value):
    """When no outbreaks exist, sensitivity has no rows."""
    observations = pd.DataFrame([{"location": "A", "time_period": "2023-06-15", "disease_cases": threshold_value - 10}])
    forecasts = _make_forecasts("A", "2023-06-15", 1, [threshold_value - 5] * 10)
    metric = SensitivityMetric(historical_observations=historical_observations)
    result = metric.get_detailed_metric(observations, forecasts)
    assert len(result) == 0


def test_specificity_no_non_outbreaks(historical_observations, threshold_value):
    """When all periods are outbreaks, specificity has no rows."""
    observations = pd.DataFrame([{"location": "A", "time_period": "2023-06-15", "disease_cases": threshold_value + 10}])
    forecasts = _make_forecasts("A", "2023-06-15", 1, [threshold_value + 5] * 10)
    metric = SpecificityMetric(historical_observations=historical_observations)
    result = metric.get_detailed_metric(observations, forecasts)
    assert len(result) == 0


def test_sensitivity_empty_without_historical_observations():
    """SensitivityMetric returns empty result when historical_observations is not provided."""
    observations = pd.DataFrame([{"location": "A", "time_period": "2023-06-15", "disease_cases": 200.0}])
    forecasts = _make_forecasts("A", "2023-06-15", 1, [150.0] * 10)
    metric = SensitivityMetric()
    result = metric.get_detailed_metric(observations, forecasts)
    assert len(result) == 0
