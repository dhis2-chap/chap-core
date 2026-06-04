"""Unit tests for the `_log_transform_for_surrogate` helper.

Pins down the behaviour that fixes the pre-existing log1p NaN crash:
negative model outputs get clipped to 0, non-finite outputs (NaN/inf
straight from the model) drop their row from X/z/weights.
"""

import logging

import numpy as np
import pytest

from chap_core.explainability.lime import _log_transform_for_surrogate


def _arrays(y_values: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build matching X / y / weights with `len(y_values)` rows."""
    n = len(y_values)
    X = np.eye(n)
    y = np.asarray(y_values, dtype=float)
    weights = np.ones(n)
    return X, y, weights


class TestHappyPath:
    def test_all_nonnegative_keeps_every_row(self):
        X, y, weights = _arrays([0.0, 1.0, 10.0, 100.0])
        X_out, z_out, w_out = _log_transform_for_surrogate(X, y, weights)
        assert X_out.shape == X.shape
        assert z_out.shape == y.shape
        assert w_out.shape == weights.shape
        np.testing.assert_allclose(z_out, np.log1p(y))

    def test_z_equals_log1p_of_clipped_y(self):
        X, y, weights = _arrays([2.0, 5.0, 8.0])
        _, z_out, _ = _log_transform_for_surrogate(X, y, weights)
        np.testing.assert_allclose(z_out, np.log1p([2.0, 5.0, 8.0]))


class TestNegativeClipping:
    def test_negative_y_gets_clipped_to_zero(self):
        X, y, weights = _arrays([-5.0, 2.0, -0.5])
        X_out, z_out, w_out = _log_transform_for_surrogate(X, y, weights)
        # All rows survive (clip + log1p(0) = 0 is finite).
        assert X_out.shape == X.shape
        # First and third rows had negative y → z=0.
        assert z_out[0] == pytest.approx(0.0)
        assert z_out[2] == pytest.approx(0.0)
        # Middle row passed through as log1p(2).
        assert z_out[1] == pytest.approx(np.log1p(2.0))

    def test_logs_warning_when_clipping(self, caplog):
        X, y, weights = _arrays([-1.0, 0.0, 1.0])
        with caplog.at_level(logging.WARNING, logger="chap_core.explainability.lime"):
            _log_transform_for_surrogate(X, y, weights)
        assert any("clipping" in rec.message and "negative" in rec.message for rec in caplog.records)

    def test_no_warning_when_no_negatives(self, caplog):
        X, y, weights = _arrays([0.0, 1.0, 2.0])
        with caplog.at_level(logging.WARNING, logger="chap_core.explainability.lime"):
            _log_transform_for_surrogate(X, y, weights)
        assert not caplog.records


class TestNonFiniteDropping:
    def test_nan_rows_get_dropped(self):
        X, y, weights = _arrays([1.0, float("nan"), 2.0, float("nan")])
        X_out, z_out, w_out = _log_transform_for_surrogate(X, y, weights)
        # Two rows remain: the ones where y was 1.0 and 2.0.
        assert X_out.shape == (2, 4)
        assert z_out.shape == (2,)
        assert w_out.shape == (2,)
        np.testing.assert_allclose(z_out, np.log1p([1.0, 2.0]))

    def test_inf_rows_get_dropped(self):
        X, y, weights = _arrays([1.0, float("inf"), 2.0])
        X_out, z_out, w_out = _log_transform_for_surrogate(X, y, weights)
        # log1p(inf) = inf, which is non-finite -> dropped.
        assert X_out.shape == (2, 3)
        np.testing.assert_allclose(z_out, np.log1p([1.0, 2.0]))

    def test_logs_warning_when_dropping(self, caplog):
        X, y, weights = _arrays([1.0, float("nan"), 2.0])
        with caplog.at_level(logging.WARNING, logger="chap_core.explainability.lime"):
            _log_transform_for_surrogate(X, y, weights)
        assert any("dropping" in rec.message and "non-finite" in rec.message for rec in caplog.records)

    def test_all_non_finite_raises_value_error(self):
        X, y, weights = _arrays([float("nan"), float("nan"), float("nan")])
        with pytest.raises(ValueError, match="All perturbed predictions were non-finite"):
            _log_transform_for_surrogate(X, y, weights)


class TestCombinedNegativeAndNanInput:
    def test_clip_first_then_drop(self):
        # Mix of: negative (clipped), normal, NaN (dropped), inf (dropped).
        X, y, weights = _arrays([-10.0, 5.0, float("nan"), float("inf")])
        X_out, z_out, w_out = _log_transform_for_surrogate(X, y, weights)
        assert X_out.shape == (2, 4)
        # Row 0 was clipped to 0 -> log1p(0) = 0; row 1 kept as log1p(5).
        np.testing.assert_allclose(z_out, [0.0, np.log1p(5.0)])
        # Weights for the kept rows are the original 1.0 ones.
        np.testing.assert_allclose(w_out, [1.0, 1.0])
