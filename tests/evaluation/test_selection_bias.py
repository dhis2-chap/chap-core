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
    estimate = estimate_selection_bias(MAEMetric(), _scores([[0.5, 0.8]] * 6), future_splits=12)

    assert estimate.final_index == 0
    np.testing.assert_allclose(estimate.final_values, [0.5, 0.2])
    np.testing.assert_allclose([estimate.estimated_bias, estimate.mean_optimism], [0, 0], atol=1e-12)
    # 6 splits: windows [a, b) with b < 6, each scored on the splits from b on.
    assert estimate.n_pairs == sum((6 - b) * b for b in range(1, 6))
    assert [p.x for p in estimate.window_size_points] == [1, 2, 3, 4, 5]
    assert [p.x for p in estimate.distance_points] == [0, 1, 2, 3, 4]
    assert sum(p.n_pairs for p in estimate.window_size_points) == estimate.n_pairs


def test_models_taking_turns_flatter_short_windows_most():
    # The two models are equally good overall but alternate: a short window picks
    # whichever was lucky in it, and that luck does not carry over to later splits.
    estimate = estimate_selection_bias(MAEMetric(), _scores([[0.2, 0.8], [0.8, 0.2]] * 6), future_splits=12)

    window_size = {p.x: p.value for p in estimate.window_size_points}
    assert window_size[1] > window_size[2] > 0
    assert estimate.coefficients[1] > 0
    assert estimate.mean_optimism > estimate.estimated_bias
    for point in estimate.window_size_points:
        a0, a1, _ = estimate.coefficients
        assert point.fitted == pytest.approx(a0 + a1 / np.sqrt(point.x))


def test_too_few_split_points_are_rejected():
    with pytest.raises(SelectionBiasError):
        estimate_selection_bias(MAEMetric(), _scores([[0.5, 0.8]] * 2), future_splits=12)


def test_split_scores_sum_each_split_on_the_forecasts_every_model_made():
    periods = ["2023-02", "2023-03", "2023-04"]
    observations = pd.DataFrame(
        {"location": ["a"] * 3 + ["b"] * 3, "time_period": periods * 2, "disease_cases": [10.0] * 6}
    )

    def forecasts(error, locations=("a", "b")):
        return pd.DataFrame(
            [
                {"location": loc, "time_period": p, "horizon_distance": h, "sample": 0, "forecast": 10.0 + error}
                for loc in locations
                for p in periods
                for h in (1, 2)
            ]
        )

    # The second model has no forecasts for org unit b, so b is left out for every model.
    split_periods, scores = split_scores(
        MAEMetric(), observations, [forecasts(1), forecasts(2, ("a",)), forecasts(4)], horizon_distance=1
    )

    # Horizon 1 forecasts of February to April were made from January to March.
    assert split_periods == ["202301", "202302", "202303"]
    np.testing.assert_allclose(scores.model_sums, [[1, 2]] * 3)
    np.testing.assert_allclose(scores.baseline_sums, [4] * 3)
    np.testing.assert_allclose(scores.counts, [1] * 3)
