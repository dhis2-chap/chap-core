import numpy as np
import pandas as pd
import pytest

from chap_core.assessment.metrics.mae import MAEMetric
from chap_core.services.selection_bias_service import (
    SelectionBiasError,
    SplitScores,
    estimate_selection_bias,
    split_scores,
)


def _scores(model_errors):
    """Per-split scores with one forecast per split and a baseline that always errs by 1."""
    model_errors = np.asarray(model_errors, float)
    n = len(model_errors)
    return SplitScores(model_sums=model_errors, baseline_sums=np.ones(n), counts=np.ones(n))


def test_a_model_that_is_always_best_by_the_same_margin_has_no_selection_bias():
    estimate = estimate_selection_bias(MAEMetric(), _scores([[0.5, 0.8]] * 6))

    assert estimate.final_index == 0
    np.testing.assert_allclose(estimate.final_values, [0.5, 0.2])
    np.testing.assert_allclose([estimate.estimated_bias, estimate.mean_gap], [0, 0], atol=1e-12)
    # 6 splits: windows [a, b) with b < 6, each scored on the splits from b on.
    assert estimate.n_pairs == sum((6 - b) * b for b in range(1, 6))
    assert (estimate.n_windows, estimate.n_windows_other_pick) == (15, 0)
    assert [p.x for p in estimate.window_size_points] == [1, 2, 3, 4, 5]
    assert [p.x for p in estimate.distance_points] == [0, 1, 2, 3, 4]
    assert sum(p.n_pairs for p in estimate.window_size_points) == estimate.n_pairs


def test_a_model_that_wins_every_window_has_no_selection_bias_even_when_its_score_drifts():
    # Both models get worse over time, and the margin varies, but the first wins every window.
    errors = [[0.1 + 0.1 * t, 0.3 + 0.12 * t + 0.05 * (t % 2)] for t in range(8)]
    estimate = estimate_selection_bias(MAEMetric(), _scores(errors))

    assert estimate.final_index == 0
    assert estimate.n_windows_other_pick == 0
    assert estimate.estimated_bias == 0
    assert all(point.value == 0 for point in estimate.window_size_points + estimate.distance_points)


def test_a_lucky_rival_in_short_windows_gives_a_positive_bias():
    # The first model always errs by 0.3; the second by 0.1 and 0.6 in turn, 0.35 on average. One-split windows
    # that land on a 0.1 pick the second model, which is worse than the final pick on the later splits.
    errors = [[0.3, 0.1 if t % 2 == 0 else 0.6] for t in range(12)]
    estimate = estimate_selection_bias(MAEMetric(), _scores(errors))

    assert estimate.final_index == 0
    assert estimate.n_windows_other_pick > 0
    window_size = {p.x: p.value for p in estimate.window_size_points}
    assert window_size[1] > 0
    assert window_size[2] == 0  # two splits always average 0.35, so they pick the first model
    assert estimate.inverse_sqrt_k > 0
    assert estimate.mean_gap > estimate.estimated_bias > 0
    for point in estimate.window_size_points:
        assert point.fitted == pytest.approx(estimate.inverse_sqrt_k / np.sqrt(point.x))


def test_too_few_split_points_are_rejected():
    with pytest.raises(SelectionBiasError):
        estimate_selection_bias(MAEMetric(), _scores([[0.5, 0.8]] * 2))


def test_test_splits_start_after_the_gap():
    # 5 splits with a gap of 1: windows [a, b) with b + 1 < 5, each scored on the splits from b + 1 on.
    estimate = estimate_selection_bias(MAEMetric(), _scores([[0.5, 0.8]] * 5), gap_splits=1)

    assert estimate.n_pairs == sum((5 - b - 1) * b for b in range(1, 4)) == 10
    assert [p.x for p in estimate.window_size_points] == [1, 2, 3]
    assert [p.x for p in estimate.distance_points] == [0, 1, 2]


def test_too_few_split_points_for_the_gap_are_rejected():
    with pytest.raises(SelectionBiasError, match="at least 5 split points"):
        estimate_selection_bias(MAEMetric(), _scores([[0.5, 0.8]] * 4), gap_splits=2)


PERIODS = ["2023-02", "2023-03", "2023-04"]
OBSERVATIONS = pd.DataFrame(
    {"location": ["a"] * 3 + ["b"] * 3, "time_period": PERIODS * 2, "disease_cases": [10.0] * 6}
)


def _forecasts(error, locations=("a", "b")):
    return pd.DataFrame(
        [
            {"location": loc, "time_period": p, "horizon_distance": h, "sample": 0, "forecast": 10.0 + error}
            for loc in locations
            for p in PERIODS
            for h in (1, 2)
        ]
    )


def test_split_scores_sum_each_split_on_the_forecasts_every_model_made():
    # The second model has no forecasts for org unit b, so b is left out for every model.
    split_periods, horizons, scores = split_scores(
        MAEMetric(), OBSERVATIONS, [_forecasts(1), _forecasts(2, ("a",)), _forecasts(4)], horizon_distance=1
    )

    # Horizon 1 forecasts of February to April were made from January to March.
    assert horizons == [1]
    assert split_periods == ["202301", "202302", "202303"]
    np.testing.assert_allclose(scores.model_sums, [[1, 2]] * 3)
    np.testing.assert_allclose(scores.baseline_sums, [4] * 3)
    np.testing.assert_allclose(scores.counts, [1] * 3)


def test_split_scores_pool_every_horizon_distance_by_default():
    split_periods, horizons, scores = split_scores(
        MAEMetric(), OBSERVATIONS, [_forecasts(1), _forecasts(2, ("a",)), _forecasts(4)]
    )

    # Horizon 2 forecasts of February to April were made from December to February.
    assert horizons == [1, 2]
    assert split_periods == ["202212", "202301", "202302", "202303"]
    np.testing.assert_allclose(scores.counts, [1, 2, 2, 1])
    np.testing.assert_allclose(scores.model_sums, [[1, 2], [2, 4], [2, 4], [1, 2]])
    np.testing.assert_allclose(scores.baseline_sums, [4, 8, 8, 4])
