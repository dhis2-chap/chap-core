"""Unit tests for perturbation samplers."""

import random

import numpy as np
import pandas as pd

from chap_core.explainability.perturb import (
    ConstantTransform,
    GlobalMean,
    LinearInterpolation,
    LocalMean,
    RandomUniform,
)


def _hist(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"x": values})


class TestLinearInterpolation:
    def test_interpolates_between_boundary_values(self):
        # Replacement of indices 2..4 should interpolate linearly from x[1]=10 to x[5]=50.
        hist = _hist([0.0, 10.0, 999.0, 999.0, 999.0, 50.0, 60.0])
        sampler = LinearInterpolation(rng=random.Random(0))
        out = sampler.sample(hist, indices=(2, 5), feature_name="x", length=3)
        assert len(out) == 3
        # Slope from 10 -> 50 over 5 positions; the inner three should be 20, 30, 40.
        assert out == [20.0, 30.0, 40.0]

    def test_nan_boundary_falls_back_to_zeros(self):
        hist = _hist([float("nan"), 1.0, 2.0, 3.0])
        out = LinearInterpolation(rng=random.Random(0)).sample(hist, indices=(0, 2), feature_name="x", length=2)
        assert out == [0.0, 0.0]


class TestConstantTransform:
    def test_always_returns_zeros_of_requested_length(self):
        out = ConstantTransform(rng=random.Random(0)).sample(
            _hist([1.0, 2.0, 3.0]), indices=(0, 3), feature_name="x", length=5
        )
        assert out == [0.0, 0.0, 0.0, 0.0, 0.0]


class TestLocalMean:
    def test_returns_mean_of_segment_repeated(self):
        hist = _hist([2.0, 4.0, 6.0, 8.0, 10.0])
        out = LocalMean(rng=random.Random(0)).sample(hist, indices=(1, 4), feature_name="x", length=3)
        # Mean of [4, 6, 8] = 6.0
        assert out == [6.0, 6.0, 6.0]

    def test_all_nan_segment_falls_back_to_zeros(self):
        hist = _hist([1.0, float("nan"), float("nan"), 4.0])
        out = LocalMean(rng=random.Random(0)).sample(hist, indices=(1, 3), feature_name="x", length=2)
        assert out == [0.0, 0.0]


class TestGlobalMean:
    def test_returns_mean_of_full_series_repeated(self):
        hist = _hist([1.0, 2.0, 3.0, 4.0, 5.0])
        out = GlobalMean(rng=random.Random(0)).sample(hist, indices=(1, 3), feature_name="x", length=2)
        assert out == [3.0, 3.0]


class TestRandomUniform:
    def test_values_stay_within_observed_range(self):
        dataset = _hist([1.0, 5.0, 10.0])
        sampler = RandomUniform(rng=random.Random(42), dataset=dataset)
        out = sampler.sample(_hist([7.0]), indices=(0, 1), feature_name="x", length=20)
        arr = np.asarray(out)
        assert arr.shape == (20,)
        assert arr.min() >= 1.0
        assert arr.max() <= 10.0

    def test_seeded_rng_is_reproducible(self):
        dataset = _hist([0.0, 100.0])
        a = RandomUniform(rng=random.Random(7), dataset=dataset).sample(
            _hist([0.0]), indices=(0, 1), feature_name="x", length=5
        )
        b = RandomUniform(rng=random.Random(7), dataset=dataset).sample(
            _hist([0.0]), indices=(0, 1), feature_name="x", length=5
        )
        assert a == b
