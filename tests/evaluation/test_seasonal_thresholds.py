import numpy as np
import pandas as pd
import pytest

from chap_core.assessment.thresholds.seasonal import compute_seasonal_thresholds


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


@pytest.mark.parametrize(
    "time_period_format",
    [
        lambda year: f"{year}-06-15",
        lambda year: f"{year}-06-01",
        lambda year: f"{year}-06",
        lambda year: f"{year}06",
    ],
)
def test_compute_seasonal_thresholds_period_formats(time_period_format, threshold_value):
    """Vectorised month extraction matches per-row TimePeriod.parse across monthly string formats."""
    values = [90.0, 95.0, 100.0, 105.0, 110.0]
    rows = [
        {"location": "A", "time_period": time_period_format(year), "disease_cases": val}
        for year, val in zip(range(2018, 2023), values)
    ]
    result = compute_seasonal_thresholds(pd.DataFrame(rows))
    assert len(result) == 1
    assert result.iloc[0]["month"] == 6
    assert result.iloc[0]["threshold"] == pytest.approx(threshold_value)


def test_compute_seasonal_thresholds_single_datapoint_is_nan():
    """A (location, month) with a single historical value has undefined std, so the threshold is NaN (-> null)."""
    df = pd.DataFrame(
        [
            {"location": "A", "time_period": "2020-06", "disease_cases": 100.0},
            {"location": "B", "time_period": "2020-06", "disease_cases": 50.0},
            {"location": "B", "time_period": "2021-06", "disease_cases": 70.0},
        ]
    )
    result = compute_seasonal_thresholds(df)
    a = result[result["location"] == "A"].iloc[0]
    b = result[result["location"] == "B"].iloc[0]
    assert np.isnan(a["threshold"])  # single data point -> cannot compute -> NaN
    assert not np.isnan(b["threshold"])  # two data points -> computed
