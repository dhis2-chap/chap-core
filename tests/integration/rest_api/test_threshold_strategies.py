"""Tests for the threshold strategy registry and the seasonal and percentile strategies."""

import pytest

from chap_core.assessment.thresholds import get_threshold_strategy, list_threshold_strategies, threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase
from chap_core.assessment.thresholds.params import PercentileParams, SeasonalParams
from chap_core.assessment.thresholds.seasonal import compute_seasonal_thresholds
from chap_core.spatio_temporal_data.converters import observations_to_dataframe


def _disease_cases_df(dataset_observations):
    df = observations_to_dataframe(dataset_observations)
    df = df[df["feature_name"] == "disease_cases"].rename(columns={"value": "disease_cases"})
    return df[["location", "time_period", "disease_cases"]]


def _strategy(strategy_id):
    strategy_cls = get_threshold_strategy(strategy_id)
    assert strategy_cls is not None
    return strategy_cls()


def test_strategies_are_registered():
    ids = {s["id"] for s in list_threshold_strategies()}
    assert {"seasonal", "percentile"}.issubset(ids)


def test_unknown_strategy_returns_none():
    assert get_threshold_strategy("does_not_exist") is None


def test_registering_mismatched_params_literal_raises():
    with pytest.raises(ValueError, match="does not match"):

        @threshold("not_seasonal", "Mismatched", SeasonalParams)
        class Mismatched(ThresholdStrategyBase):
            def compute(self, historical_observations, period_ids, params):
                raise NotImplementedError

    assert get_threshold_strategy("not_seasonal") is None


def test_registered_params_models_match_strategy_ids():
    for strategy_id in (s["id"] for s in list_threshold_strategies()):
        cls = get_threshold_strategy(strategy_id)
        assert cls is not None
        assert cls.params_model.model_fields["type"].default == strategy_id


def test_seasonal_strategy_shape(dataset_observations, org_units):
    df = _disease_cases_df(dataset_observations)
    period_ids = ["2023-01", "2023-02"]
    result = _strategy("seasonal").compute(df, period_ids, SeasonalParams())
    assert set(result.columns) == {"period_id", "location", "line", "threshold"}
    assert len(result) == len(period_ids) * len(org_units)
    assert set(result["period_id"]) == set(period_ids)
    assert set(result["location"]) == set(org_units)
    assert set(result["line"]) == {0}


def test_seasonal_strategy_parity_with_compute_seasonal_thresholds(dataset_observations):
    df = _disease_cases_df(dataset_observations)
    result = _strategy("seasonal").compute(df, ["2023-01"], SeasonalParams())
    per_month = compute_seasonal_thresholds(df)
    january = per_month[per_month["month"] == 1]
    for row in result.itertuples():
        expected = january[january["location"] == row.location]["threshold"].iloc[0]
        assert row.threshold == expected


def test_seasonal_strategy_multi_line(endemic_channel_observations):
    result = _strategy("seasonal").compute(
        endemic_channel_observations, ["2023-01"], SeasonalParams(std_multiplier=[1.0, 2.0])
    )
    assert set(result["line"]) == {0, 1}
    for location in ("loc_1", "loc_2"):
        rows = result[result["location"] == location].set_index("line")["threshold"]
        single = compute_seasonal_thresholds(endemic_channel_observations, k=1.0)
        expected = single[(single["location"] == location) & (single["month"] == 1)]["threshold"].iloc[0]
        assert rows[0] == expected
        assert rows[0] < rows[1]


def test_seasonal_strategy_weekly(dataset_observations_weekly, org_units):
    df = _disease_cases_df(dataset_observations_weekly)
    period_ids = ["2023W01", "2023W02"]
    result = _strategy("seasonal").compute(df, period_ids, SeasonalParams())
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
    strategy = _strategy("seasonal")
    padded = strategy.compute(df, ["2023W01"], SeasonalParams()).set_index("location")["threshold"]
    unpadded = strategy.compute(df, ["2023W1"], SeasonalParams()).set_index("location")["threshold"]
    assert padded.equals(unpadded)


def test_seasonal_strategy_frequency_mismatch_raises(dataset_observations_weekly):
    df = _disease_cases_df(dataset_observations_weekly)
    with pytest.raises(ValueError, match="frequency"):
        _strategy("seasonal").compute(df, ["2023-01"], SeasonalParams())


def test_percentile_strategy_values(endemic_channel_observations):
    result = _strategy("percentile").compute(endemic_channel_observations, ["2023-01"], PercentileParams(quantile=0.75))
    expected = (
        endemic_channel_observations[endemic_channel_observations["time_period"].str.endswith("-01")]
        .groupby("location")["disease_cases"]
        .quantile(0.75)
    )
    for row in result.itertuples():
        assert row.threshold == expected[row.location]


def test_percentile_strategy_multi_line_order(endemic_channel_observations):
    result = _strategy("percentile").compute(
        endemic_channel_observations, ["2023-01"], PercentileParams(quantile=[0.75, 0.25])
    )
    assert set(result["line"]) == {0, 1}
    for location in ("loc_1", "loc_2"):
        rows = result[result["location"] == location].set_index("line")["threshold"]
        # line order follows the request order, not sorted quantile order
        assert rows[0] > rows[1]


def test_percentile_strategy_baseline_window(endemic_channel_observations):
    """A 2-year baseline anchored at 2023 uses only 2021-2022 observations."""
    strategy = _strategy("percentile")
    windowed = strategy.compute(
        endemic_channel_observations, ["2023-01"], PercentileParams(quantile=0.5, baseline_years=2)
    )
    expected = (
        endemic_channel_observations[endemic_channel_observations["time_period"].isin(["2021-01", "2022-01"])]
        .groupby("location")["disease_cases"]
        .median()
    )
    for row in windowed.itertuples():
        assert row.threshold == expected[row.location]


def test_percentile_strategy_all_history_with_null_baseline(endemic_channel_observations):
    result = _strategy("percentile").compute(
        endemic_channel_observations, ["2023-01"], PercentileParams(quantile=0.5, baseline_years=None)
    )
    expected = (
        endemic_channel_observations[endemic_channel_observations["time_period"].str.endswith("-01")]
        .groupby("location")["disease_cases"]
        .median()
    )
    for row in result.itertuples():
        assert row.threshold == expected[row.location]


def test_percentile_strategy_empty_window_raises(endemic_channel_observations):
    with pytest.raises(ValueError, match="No observations"):
        _strategy("percentile").compute(endemic_channel_observations, ["2050-01"], PercentileParams())


def test_percentile_params_validation():
    with pytest.raises(ValueError):
        PercentileParams(quantile=1.5)
    with pytest.raises(ValueError):
        PercentileParams(quantile=[0.5, -0.1])
    with pytest.raises(ValueError):
        PercentileParams(quantile=[])
    with pytest.raises(ValueError):
        PercentileParams(baseline_years=0)
