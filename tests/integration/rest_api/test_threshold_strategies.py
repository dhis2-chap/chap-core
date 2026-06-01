"""Tests for the threshold strategy registry and the seasonal strategy."""

from chap_core.assessment.thresholds import get_threshold_strategy, list_threshold_strategies
from chap_core.assessment.thresholds.seasonal import compute_seasonal_thresholds
from chap_core.spatio_temporal_data.converters import observations_to_dataframe


def _disease_cases_df(dataset_observations):
    df = observations_to_dataframe(dataset_observations)
    df = df[df["feature_name"] == "disease_cases"].rename(columns={"value": "disease_cases"})
    return df[["location", "time_period", "disease_cases"]]


def _seasonal_strategy():
    strategy_cls = get_threshold_strategy("seasonal")
    assert strategy_cls is not None
    return strategy_cls()


def test_seasonal_strategy_is_registered():
    assert "seasonal" in {s["id"] for s in list_threshold_strategies()}
    assert get_threshold_strategy("seasonal") is not None


def test_unknown_strategy_returns_none():
    assert get_threshold_strategy("does_not_exist") is None


def test_seasonal_strategy_shape(dataset_observations, org_units):
    df = _disease_cases_df(dataset_observations)
    period_ids = ["2023-01", "2023-02"]
    result = _seasonal_strategy().compute(df, period_ids)
    assert set(result.columns) == {"period_id", "location", "threshold"}
    assert len(result) == len(period_ids) * len(org_units)
    assert set(result["period_id"]) == set(period_ids)
    assert set(result["location"]) == set(org_units)


def test_seasonal_strategy_parity_with_compute_seasonal_thresholds(dataset_observations):
    df = _disease_cases_df(dataset_observations)
    result = _seasonal_strategy().compute(df, ["2023-01"])
    per_month = compute_seasonal_thresholds(df)
    january = per_month[per_month["month"] == 1]
    for row in result.itertuples():
        expected = january[january["location"] == row.location]["threshold"].iloc[0]
        assert row.threshold == expected
