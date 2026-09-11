"""Tests for the threshold strategy registry and the built-in endemic channel strategies."""

from typing import get_args

import numpy as np
import pandas as pd
import pytest

from chap_core.assessment.thresholds import (
    get_threshold_strategy,
    list_threshold_strategies,
    params_type_literal,
    threshold,
)
from chap_core.assessment.thresholds.base import ThresholdStrategyBase
from chap_core.assessment.thresholds.geometric import compute_geometric_thresholds
from chap_core.assessment.thresholds.params import (
    GeometricParams,
    PercentileParams,
    SeasonalParams,
    ThresholdParams,
)
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
    assert {"seasonal", "percentile", "geometric"}.issubset(ids)


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
        assert params_type_literal(cls.params_model) == strategy_id


def test_builtin_strategies_are_in_params_union():
    """Every strategy shipped with chap_core must be a member of ThresholdParams, and vice versa.

    Otherwise GET /thresholds/strategies advertises a strategy that POST /thresholds rejects
    with a 422. Strategies registered from outside the package (e.g. the contributor guide's
    example) are excluded, since adding them to the union is a separate step.
    """
    union, _ = get_args(ThresholdParams)
    union_ids = {params_type_literal(model) for model in get_args(union)}
    builtin_ids = {
        strategy_id
        for strategy_id in (s["id"] for s in list_threshold_strategies())
        if get_threshold_strategy(strategy_id).__module__.startswith("chap_core.assessment.thresholds")
    }
    assert builtin_ids == union_ids


def test_seasonal_strategy_shape(dataset_observations, org_units):
    df = _disease_cases_df(dataset_observations)
    period_ids = ["2023-01", "2023-02"]
    result = _strategy("seasonal").compute(df, period_ids, SeasonalParams(type="seasonal"))
    assert set(result.columns) == {"period_id", "location", "line", "threshold"}
    assert len(result) == len(period_ids) * len(org_units)
    assert set(result["period_id"]) == set(period_ids)
    assert set(result["location"]) == set(org_units)
    assert set(result["line"]) == {0}


def test_seasonal_strategy_parity_with_compute_seasonal_thresholds(dataset_observations):
    df = _disease_cases_df(dataset_observations)
    result = _strategy("seasonal").compute(df, ["2023-01"], SeasonalParams(type="seasonal"))
    per_month = compute_seasonal_thresholds(df)
    january = per_month[per_month["month"] == 1]
    for row in result.itertuples():
        expected = january[january["location"] == row.location]["threshold"].iloc[0]
        assert row.threshold == expected


def test_seasonal_strategy_multi_line(endemic_channel_observations):
    result = _strategy("seasonal").compute(
        endemic_channel_observations, ["2023-01"], SeasonalParams(type="seasonal", std_multiplier=[1.0, 2.0])
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
    result = _strategy("seasonal").compute(df, period_ids, SeasonalParams(type="seasonal"))
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
    padded = strategy.compute(df, ["2023W01"], SeasonalParams(type="seasonal")).set_index("location")["threshold"]
    unpadded = strategy.compute(df, ["2023W1"], SeasonalParams(type="seasonal")).set_index("location")["threshold"]
    assert padded.equals(unpadded)


def test_seasonal_strategy_frequency_mismatch_raises(dataset_observations_weekly):
    df = _disease_cases_df(dataset_observations_weekly)
    with pytest.raises(ValueError, match="frequency"):
        _strategy("seasonal").compute(df, ["2023-01"], SeasonalParams(type="seasonal"))


def test_percentile_strategy_values(endemic_channel_observations):
    result = _strategy("percentile").compute(
        endemic_channel_observations, ["2023-01"], PercentileParams(type="percentile", quantile=0.75)
    )
    expected = (
        endemic_channel_observations[endemic_channel_observations["time_period"].str.endswith("-01")]
        .groupby("location")["disease_cases"]
        .quantile(0.75)
    )
    for row in result.itertuples():
        assert row.threshold == expected[row.location]


def test_percentile_strategy_multi_line_order(endemic_channel_observations):
    result = _strategy("percentile").compute(
        endemic_channel_observations, ["2023-01"], PercentileParams(type="percentile", quantile=[0.75, 0.25])
    )
    assert set(result["line"]) == {0, 1}
    for location in ("loc_1", "loc_2"):
        rows = result[result["location"] == location].set_index("line")["threshold"]
        # line order follows the request order, not sorted quantile order
        assert rows[0] > rows[1]


def test_percentile_strategy_baseline_window(endemic_channel_observations):
    """A 2-year baseline over data ending in 2022 uses only 2021-2022 observations."""
    strategy = _strategy("percentile")
    windowed = strategy.compute(
        endemic_channel_observations, ["2023-01"], PercentileParams(type="percentile", quantile=0.5, baseline_years=2)
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
        endemic_channel_observations,
        ["2023-01"],
        PercentileParams(type="percentile", quantile=0.5, baseline_years=None),
    )
    expected = (
        endemic_channel_observations[endemic_channel_observations["time_period"].str.endswith("-01")]
        .groupby("location")["disease_cases"]
        .median()
    )
    for row in result.itertuples():
        assert row.threshold == expected[row.location]


def test_percentile_strategy_is_static_across_requested_periods(endemic_channel_observations):
    """Past, in-range and future periods of the same season get the same line."""
    strategy = _strategy("percentile")
    params = PercentileParams(type="percentile", quantile=0.75, baseline_years=3)
    separate = [
        strategy.compute(endemic_channel_observations, [period], params).set_index("location")["threshold"]
        for period in ("2019-01", "2022-01", "2030-01")
    ]
    combined = strategy.compute(endemic_channel_observations, ["2019-01", "2022-01", "2030-01"], params)
    for single in separate[1:]:
        pd.testing.assert_series_equal(single, separate[0])
    for row in combined.itertuples():
        assert row.threshold == separate[0][row.location]


def test_percentile_strategy_excludes_partial_final_year(
    endemic_channel_observations, endemic_channel_observations_partial_year
):
    """An in-progress final year is not part of the baseline, so it cannot raise its own threshold."""
    strategy = _strategy("percentile")
    params = PercentileParams(type="percentile", quantile=0.75, baseline_years=2)
    complete = strategy.compute(endemic_channel_observations, ["2023-03"], params)
    with_partial = strategy.compute(endemic_channel_observations_partial_year, ["2023-03"], params)
    pd.testing.assert_frame_equal(with_partial, complete)


def test_percentile_strategy_no_complete_year_raises(endemic_channel_observations_partial_year):
    only_partial = endemic_channel_observations_partial_year[
        endemic_channel_observations_partial_year["time_period"].str.startswith("2023")
    ]
    with pytest.raises(ValueError, match="No complete years"):
        _strategy("percentile").compute(only_partial, ["2023-03"], PercentileParams(type="percentile"))


def test_params_lines_follow_request_order():
    assert SeasonalParams(type="seasonal").lines == [2.0]
    assert SeasonalParams(type="seasonal", std_multiplier=[1.0, 2.0]).lines == [1.0, 2.0]
    assert PercentileParams(type="percentile").lines == [0.75]
    assert PercentileParams(type="percentile", quantile=[0.75, 0.25]).lines == [0.75, 0.25]


def test_params_type_is_required():
    with pytest.raises(ValueError):
        PercentileParams()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        SeasonalParams()  # type: ignore[call-arg]


def test_percentile_params_validation():
    with pytest.raises(ValueError):
        PercentileParams(type="percentile", quantile=1.5)
    with pytest.raises(ValueError):
        PercentileParams(type="percentile", quantile=[0.5, -0.1])
    with pytest.raises(ValueError):
        PercentileParams(type="percentile", quantile=[])
    with pytest.raises(ValueError):
        PercentileParams(type="percentile", baseline_years=0)


def test_geometric_strategy_shape(dataset_observations, org_units):
    df = _disease_cases_df(dataset_observations)
    period_ids = ["2023-01", "2023-02"]
    result = _strategy("geometric").compute(df, period_ids, GeometricParams(type="geometric"))
    assert set(result.columns) == {"period_id", "location", "line", "threshold"}
    assert len(result) == len(period_ids) * len(org_units)
    assert set(result["period_id"]) == set(period_ids)
    assert set(result["location"]) == set(org_units)
    assert set(result["line"]) == {0}


def test_geometric_strategy_weekly(dataset_observations_weekly, org_units):
    df = _disease_cases_df(dataset_observations_weekly)
    period_ids = ["2023W01", "2023W02"]
    result = _strategy("geometric").compute(df, period_ids, GeometricParams(type="geometric"))
    assert len(result) == len(period_ids) * len(org_units)
    assert result["threshold"].notna().all()


def test_geometric_strategy_frequency_mismatch_raises(dataset_observations_weekly):
    df = _disease_cases_df(dataset_observations_weekly)
    with pytest.raises(ValueError, match="frequency"):
        _strategy("geometric").compute(df, ["2023-01"], GeometricParams(type="geometric"))


def test_geometric_strategy_values(endemic_channel_observations):
    result = _strategy("geometric").compute(
        endemic_channel_observations, ["2023-01"], GeometricParams(type="geometric")
    )
    january = endemic_channel_observations[endemic_channel_observations["time_period"].str.endswith("-01")]
    for row in result.itertuples():
        logged = np.log1p(january[january["location"] == row.location]["disease_cases"])
        expected = np.expm1(logged.mean() + 2.0 * logged.std(ddof=1))
        assert row.threshold == pytest.approx(expected)


def test_geometric_strategy_parity_with_compute_geometric_thresholds(endemic_channel_observations):
    result = _strategy("geometric").compute(
        endemic_channel_observations, ["2023-01"], GeometricParams(type="geometric")
    )
    per_month = compute_geometric_thresholds(endemic_channel_observations)
    january = per_month[per_month["month"] == 1]
    for row in result.itertuples():
        assert row.threshold == january[january["location"] == row.location]["threshold"].iloc[0]


def test_geometric_strategy_handles_zero_counts(endemic_channel_observations):
    """log1p keeps zero counts, so an all-zero season gives a finite threshold of zero."""
    zeroed = endemic_channel_observations.assign(disease_cases=0.0)
    result = _strategy("geometric").compute(zeroed, ["2023-01"], GeometricParams(type="geometric"))
    assert np.isfinite(result["threshold"]).all()
    assert result["threshold"].eq(0.0).all()


def test_geometric_centre_is_below_arithmetic_after_epidemic_year(endemic_channel_observations_with_epidemic):
    """With no spread term the channel is the geometric mean, which AM-GM keeps below the arithmetic mean.

    Only the centre is guaranteed to be lower. The k=2 band is not: the log-scale standard
    deviation is a relative spread, so on this five-year baseline one epidemic January makes
    the back-transformed band wider than mean + 2*std rather than tighter.
    """
    geometric = _strategy("geometric").compute(
        endemic_channel_observations_with_epidemic, ["2023-01"], GeometricParams(type="geometric", std_multiplier=0.0)
    )
    arithmetic = _strategy("seasonal").compute(
        endemic_channel_observations_with_epidemic, ["2023-01"], SeasonalParams(type="seasonal", std_multiplier=0.0)
    )
    assert (geometric.set_index("location")["threshold"] < arithmetic.set_index("location")["threshold"]).all()


def test_geometric_strategy_multi_line(endemic_channel_observations):
    result = _strategy("geometric").compute(
        endemic_channel_observations, ["2023-01"], GeometricParams(type="geometric", std_multiplier=[1.0, 3.0])
    )
    assert set(result["line"]) == {0, 1}
    for location in ("loc_1", "loc_2"):
        rows = result[result["location"] == location].set_index("line")["threshold"]
        assert rows[0] < rows[1]


def test_geometric_params_lines_follow_request_order():
    assert GeometricParams(type="geometric").lines == [2.0]
    assert GeometricParams(type="geometric", std_multiplier=[3.0, 1.0]).lines == [3.0, 1.0]


def test_geometric_params_reject_empty_line_list():
    with pytest.raises(ValueError):
        GeometricParams(type="geometric", std_multiplier=[])
