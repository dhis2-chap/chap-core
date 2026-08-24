"""Tests for the threshold strategy registry and the seasonal strategy."""

import pandas as pd
import pytest

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


def test_seasonal_strategy_weekly(dataset_observations_weekly, org_units):
    df = _disease_cases_df(dataset_observations_weekly)
    period_ids = ["2023W01", "2023W02"]
    result = _seasonal_strategy().compute(df, period_ids)
    assert set(result.columns) == {"period_id", "location", "threshold"}
    assert len(result) == len(period_ids) * len(org_units)
    assert set(result["period_id"]) == set(period_ids)
    assert result["threshold"].notna().all()
    per_week = compute_seasonal_thresholds(df)
    week_one = per_week[per_week["week"] == 1]
    for row in result[result["period_id"] == "2023W01"].itertuples():
        expected = week_one[week_one["location"] == row.location]["threshold"].iloc[0]
        assert row.threshold == expected


def test_seasonal_strategy_weekly_unpadded_period_ids(dataset_observations_weekly):
    """2023W1 and 2023W01 refer to the same week and must yield identical thresholds."""
    df = _disease_cases_df(dataset_observations_weekly)
    strategy = _seasonal_strategy()
    padded = strategy.compute(df, ["2023W01"]).set_index("location")["threshold"]
    unpadded = strategy.compute(df, ["2023W1"]).set_index("location")["threshold"]
    assert padded.equals(unpadded)


def test_seasonal_strategy_frequency_mismatch_raises(dataset_observations_weekly):
    df = _disease_cases_df(dataset_observations_weekly)
    with pytest.raises(ValueError, match="frequency"):
        _seasonal_strategy().compute(df, ["2023-01"])


def _percentile_strategy():
    strategy_cls = get_threshold_strategy("percentile")
    assert strategy_cls is not None
    return strategy_cls()


def test_percentile_strategy_is_registered():
    assert "percentile" in {s["id"] for s in list_threshold_strategies()}


def test_percentile_strategy_shape(dataset_observations, org_units):
    df = _disease_cases_df(dataset_observations)
    period_ids = ["2023-01", "2023-02"]
    result = _percentile_strategy().compute(df, period_ids)
    assert set(result.columns) == {"period_id", "location", "threshold"}
    assert len(result) == len(period_ids) * len(org_units)
    assert set(result["period_id"]) == set(period_ids)
    assert set(result["location"]) == set(org_units)


def test_percentile_strategy_weekly(dataset_observations_weekly, org_units):
    df = _disease_cases_df(dataset_observations_weekly)
    period_ids = ["2023W01", "2023W02"]
    result = _percentile_strategy().compute(df, period_ids)
    assert len(result) == len(period_ids) * len(org_units)
    assert result["threshold"].notna().all()


def test_percentile_strategy_frequency_mismatch_raises(dataset_observations_weekly):
    df = _disease_cases_df(dataset_observations_weekly)
    with pytest.raises(ValueError, match="frequency"):
        _percentile_strategy().compute(df, ["2023-01"])


def test_percentile_quantile_param_selects_the_line(endemic_channel_observations):
    """The quantile param moves the threshold: lower channel line < median < upper line."""
    strategy = _percentile_strategy()

    def line(q):
        result = strategy.compute(endemic_channel_observations, ["2023-01"], {"quantile": q})
        return result.set_index("location")["threshold"]

    lower, median, upper = line(0.25), line(0.5), line(0.75)
    assert (lower < median).all()
    assert (median < upper).all()


def test_percentile_rejects_out_of_range_quantile(dataset_observations):
    df = _disease_cases_df(dataset_observations)
    with pytest.raises(ValueError, match="between 0 and 1"):
        _percentile_strategy().compute(df, ["2023-01"], {"quantile": 75})


def test_percentile_lookback_window_excludes_older_years(dataset_observations):
    """A one-year lookback from 2023 must use only 2022, not the full 2020-2022 history."""
    df = _disease_cases_df(dataset_observations)
    strategy = _percentile_strategy()

    windowed = strategy.compute(df, ["2023-01"], {"lookback_years": 1})
    only_2022 = df[df["time_period"].astype(str).str.startswith("2022")]
    expected = strategy.compute(only_2022, ["2023-01"], {"lookback_years": None})

    pd.testing.assert_series_equal(
        windowed.set_index("location")["threshold"],
        expected.set_index("location")["threshold"],
    )


def test_percentile_lookback_excludes_the_anchor_year(dataset_observations):
    """The requested period's own year is not part of its baseline."""
    df = _disease_cases_df(dataset_observations)
    strategy = _percentile_strategy()

    # Anchored on 2022, a 1-year window is 2021 alone -- 2022's own values must not contribute.
    windowed = strategy.compute(df, ["2022-01"], {"lookback_years": 1})
    only_2021 = df[df["time_period"].astype(str).str.startswith("2021")]
    expected = strategy.compute(only_2021, ["2022-01"], {"lookback_years": None})

    pd.testing.assert_series_equal(
        windowed.set_index("location")["threshold"],
        expected.set_index("location")["threshold"],
    )


def test_percentile_rejects_non_positive_lookback_years(dataset_observations):
    df = _disease_cases_df(dataset_observations)
    with pytest.raises(ValueError, match="positive number of years or null"):
        _percentile_strategy().compute(df, ["2023-01"], {"lookback_years": 0})


def test_percentile_lookback_with_no_data_in_window_raises(dataset_observations):
    df = _disease_cases_df(dataset_observations)
    with pytest.raises(ValueError, match="No observations in the"):
        _percentile_strategy().compute(df, ["2050-01"], {"lookback_years": 1})


def test_percentile_is_not_inflated_by_a_past_epidemic(endemic_channel_observations):
    """The Uganda regression: one past epidemic year must not drag the threshold far above normal.

    Reported symptom was an actual around 2,500 against a threshold around 20,000. mean + k*std
    is pulled up because the outlier inflates the standard deviation; a percentile is not.
    """
    january = endemic_channel_observations["time_period"].str.endswith("-01")
    normal_level = float(endemic_channel_observations.loc[january, "disease_cases"].max())
    epidemic = pd.DataFrame([{"location": "loc_1", "time_period": "2020-01", "disease_cases": normal_level * 10}])
    with_epidemic = pd.concat(
        [endemic_channel_observations[~january | (endemic_channel_observations["time_period"] != "2020-01")], epidemic],
        ignore_index=True,
    )

    percentile = _percentile_strategy().compute(with_epidemic, ["2023-01"])
    seasonal = _seasonal_strategy().compute(with_epidemic, ["2023-01"])

    pct = float(percentile.set_index("location").loc["loc_1", "threshold"])
    std_based = float(seasonal.set_index("location").loc["loc_1", "threshold"])

    assert pct <= normal_level * 1.5, "percentile should stay near the endemic level"
    assert std_based > pct * 3, "mean + k*std should be visibly inflated by the epidemic year"


def test_percentile_single_observation_yields_a_value():
    """mean + k*std is NaN for a single point because std is undefined; a percentile is not."""
    df = pd.DataFrame([{"location": "loc_1", "time_period": "2022-01", "disease_cases": 2500.0}])
    result = _percentile_strategy().compute(df, ["2023-01"], {"lookback_years": None})
    assert result["threshold"].iloc[0] == 2500.0
