"""Unit tests for distance/weighter classes."""

import numpy as np

from chap_core.explainability.distance import DTW, Pairwise


class TestPairwise:
    def test_identical_row_gets_weight_one(self):
        x0 = np.array([1.0, 2.0, 3.0])
        weighter = Pairwise(kernel_width=1)
        weights = weighter.get_weights(x0.reshape(1, -1), x0)
        assert weights.shape == (1,)
        assert np.isclose(weights[0], 1.0)

    def test_distance_zero_means_max_weight(self):
        x0 = np.array([0.0, 0.0])
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        weighter = Pairwise(kernel_width=1)
        weights = weighter.get_weights(X, x0)
        # Identical row has weight 1; farther rows have smaller weights.
        assert weights[0] >= weights[1] >= weights[2]
        assert np.isclose(weights[0], 1.0)
        assert 0 < weights[2] < 1.0

    def test_kernel_width_widens_distribution(self):
        x0 = np.array([0.0])
        X = np.array([[5.0]])
        narrow = Pairwise(kernel_width=1).get_weights(X, x0)
        wide = Pairwise(kernel_width=10).get_weights(X, x0)
        # Larger kernel_width means farther points get more weight.
        assert wide[0] > narrow[0]


class TestDTW:
    """DTW operates on 2D sequences (n_timesteps x n_features) — the same shape
    ``build_dtw_sequence`` in ``lime.py`` produces from a DataFrame."""

    def test_identical_sequence_yields_unit_weight(self):
        seq = np.array([[1.0], [2.0], [3.0], [2.0], [1.0]])
        weighter = DTW(kernel_width=1.0)
        weights = weighter.get_weights([seq], seq)
        # One perturbation of the same sequence -> sigma is 0, so the
        # zero-distance branch applies and weight should be exp(0) = 1.
        assert weights.shape == (1,)
        assert np.isclose(weights[0], 1.0)

    def test_returns_finite_array(self):
        x0 = np.array([[1.0], [2.0], [3.0], [4.0]])
        perturbed = [
            np.array([[1.0], [2.0], [3.0], [4.0]]),
            np.array([[1.5], [2.5], [3.5], [4.5]]),
            np.array([[0.0], [0.0], [0.0], [0.0]]),
        ]
        weighter = DTW(kernel_width=2.0)
        weights = weighter.get_weights(perturbed, x0)
        assert weights.shape == (3,)
        assert np.all(np.isfinite(weights))

    def test_weight_is_monotonic_and_anchored_at_zero_distance(self):
        # Three perturbations at increasing DTW distance from x0; the closest
        # (identical to x0, distance 0) must get the highest weight and weight
        # must decrease monotonically. Regression for the old z-normalisation
        # which handed the max weight to the *mean*-distance perturbation.
        x0 = np.array([[0.0], [0.0], [0.0]])
        perturbed = [
            np.array([[0.0], [0.0], [0.0]]),  # distance 0
            np.array([[10.0], [10.0], [10.0]]),  # mid
            np.array([[20.0], [20.0], [20.0]]),  # far
        ]
        weights = DTW(kernel_width=1.0).get_weights(perturbed, x0)
        assert np.isclose(weights[0], 1.0), "the original (distance 0) must get the highest weight"
        assert weights[0] > weights[1] > weights[2], "weight must decrease monotonically with distance"
