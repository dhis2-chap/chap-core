import numpy as np
import pandas as pd
import pytest

from chap_core.assessment.metrics.mae import MAEMetric
from chap_core.services.selection_bias_service import (
    SelectionBiasError,
    SplitScores,
    estimate_selection_bias,
    left_out_splits,
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
    assert estimate.estimated_bias == 0
    assert estimate.block_size == 3
    assert 0 < estimate.n_draws_scored <= estimate.n_draws
    np.testing.assert_array_equal(estimate.n_draws_picked, [estimate.n_draws_scored, 0])


def test_a_model_that_wins_every_draw_has_no_selection_bias_even_when_its_score_drifts():
    # Both models get worse over time, and the margin varies, but the first is better at every split.
    errors = [[0.1 + 0.1 * t, 0.3 + 0.12 * t + 0.05 * (t % 2)] for t in range(8)]
    estimate = estimate_selection_bias(MAEMetric(), _scores(errors))

    assert estimate.final_index == 0
    assert estimate.n_draws_picked[1] == 0
    assert estimate.estimated_bias == 0


def test_a_rival_that_is_lucky_on_some_splits_gives_a_positive_bias():
    # The first model always errs by 0.3; the second by 0.1 and 0.6 in turn, 0.35 on average. A draw that
    # lands mostly on the 0.1 splits picks the second model, which is worse on the splits it left out.
    errors = [[0.3, 0.1 if t % 2 == 0 else 0.6] for t in range(12)]
    estimate = estimate_selection_bias(MAEMetric(), _scores(errors))

    assert estimate.final_index == 0
    assert 0 < estimate.n_draws_picked[1] < estimate.n_draws_picked[0]
    assert estimate.n_draws_picked.sum() == estimate.n_draws_scored
    assert estimate.estimated_bias > 0


def test_the_same_scores_give_the_same_estimate():
    scores = _scores([[0.3, 0.1 if t % 2 == 0 else 0.6] for t in range(12)])
    first = estimate_selection_bias(MAEMetric(), scores)
    second = estimate_selection_bias(MAEMetric(), scores)

    assert first.estimated_bias == second.estimated_bias
    np.testing.assert_array_equal(first.n_draws_picked, second.n_draws_picked)


def test_too_few_split_points_are_rejected():
    with pytest.raises(SelectionBiasError, match="Too few split points"):
        estimate_selection_bias(MAEMetric(), _scores([[0.5, 0.8]] * 3))


def test_left_out_splits_are_not_drawn_and_keep_the_gap_to_drawn_splits():
    drawn = np.array([True, False, False, False, False, False, True, False])

    np.testing.assert_array_equal(left_out_splits(drawn), ~drawn)
    np.testing.assert_array_equal(np.flatnonzero(left_out_splits(drawn, gap_splits=1)), [2, 3, 4])
    np.testing.assert_array_equal(np.flatnonzero(left_out_splits(drawn, gap_splits=2)), [3])


def test_a_gap_needs_more_split_points():
    scores = _scores([[0.5, 0.8]] * 12)

    assert estimate_selection_bias(MAEMetric(), scores).block_size == 3
    with pytest.raises(SelectionBiasError, match="more than 2 splits from every drawn one"):
        estimate_selection_bias(MAEMetric(), scores, gap_splits=2)
    # A gap of 3 gives blocks of 4 split points.
    assert estimate_selection_bias(MAEMetric(), _scores([[0.5, 0.8]] * 40), gap_splits=3).block_size == 4


def test_the_estimate_finds_the_bias_of_equally_good_models():
    # Six models that all err by 0.7 on average, against a baseline that errs by 1: each is 0.3 better
    # than the baseline, and whatever more the best one shows over 24 split points is selection bias.
    rng = np.random.default_rng(1)
    actual, estimated = [], []
    for table in range(100):
        estimate = estimate_selection_bias(
            MAEMetric(), _scores(rng.uniform(0.2, 1.2, size=(24, 6))), n_draws=300, seed=table
        )
        actual.append(estimate.final_values[estimate.final_index] - 0.3)
        estimated.append(estimate.estimated_bias)

    assert np.mean(actual) > 0.05
    assert np.mean(estimated) == pytest.approx(np.mean(actual), rel=0.25)


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
