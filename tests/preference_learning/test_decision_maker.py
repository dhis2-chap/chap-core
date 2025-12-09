"""Tests for DecisionMaker implementations."""

from unittest.mock import MagicMock, patch

import pytest

from chap_core.preference_learning.decision_maker import (
    DecisionMaker,
    MetricDecisionMaker,
    VisualDecisionMaker,
)


class TestDecisionMakerInterface:
    def test_decide_returns_int(self):
        """Test that decide returns an integer index."""
        metrics = [{"mae": 0.8}, {"mae": 0.5}]
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_names=["mae"],
            lower_is_better=True,
        )

        mock_evaluations = [MagicMock(), MagicMock()]
        result = decision_maker.decide(mock_evaluations)

        assert isinstance(result, int)
        assert 0 <= result < len(mock_evaluations)


class TestMetricDecisionMaker:
    def test_lower_is_better(self):
        """Test that lower values are preferred when lower_is_better=True."""
        metrics = [{"mae": 0.8}, {"mae": 0.5}]
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_names=["mae"],
            lower_is_better=True,
        )

        mock_evaluations = [MagicMock(), MagicMock()]
        preferred_index = decision_maker.decide(mock_evaluations)

        assert preferred_index == 1  # Second model has lower MAE

    def test_higher_is_better(self):
        """Test that higher values are preferred when lower_is_better=False."""
        metrics = [{"accuracy": 0.9}, {"accuracy": 0.7}]
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_names=["accuracy"],
            lower_is_better=False,
        )

        mock_evaluations = [MagicMock(), MagicMock()]
        preferred_index = decision_maker.decide(mock_evaluations)

        assert preferred_index == 0  # First model has higher accuracy

    def test_equal_values_prefers_first(self):
        """Test that equal metric values prefer first model."""
        metrics = [{"mae": 0.5}, {"mae": 0.5}]
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_names=["mae"],
            lower_is_better=True,
        )

        mock_evaluations = [MagicMock(), MagicMock()]
        preferred_index = decision_maker.decide(mock_evaluations)

        assert preferred_index == 0

    def test_multiple_candidates(self):
        """Test with more than 2 candidates."""
        metrics = [{"mae": 0.8}, {"mae": 0.3}, {"mae": 0.5}]
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_names=["mae"],
            lower_is_better=True,
        )

        mock_evaluations = [MagicMock(), MagicMock(), MagicMock()]
        preferred_index = decision_maker.decide(mock_evaluations)

        assert preferred_index == 1  # Second model has lowest MAE

    def test_mismatched_metrics_count_raises(self):
        """Test that mismatched metrics count raises error."""
        metrics = [{"mae": 0.8}]  # Only 1 metric
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_names=["mae"],
            lower_is_better=True,
        )

        mock_evaluations = [MagicMock(), MagicMock()]  # 2 evaluations

        with pytest.raises(ValueError, match="Expected 2 metrics"):
            decision_maker.decide(mock_evaluations)

    def test_fallback_to_common_metric(self):
        """Test fallback when requested metric not available."""
        metrics = [{"rmse": 0.8}, {"rmse": 0.5}]
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_names=["mae"],  # Not in metrics
            lower_is_better=True,
        )

        mock_evaluations = [MagicMock(), MagicMock()]
        preferred_index = decision_maker.decide(mock_evaluations)

        # Should fall back to rmse and pick second model
        assert preferred_index == 1

    def test_priority_ordering(self):
        """Test that first available metric in list is used."""
        metrics = [{"rmse": 0.3, "mae": 0.8}, {"rmse": 0.5, "mae": 0.2}]
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_names=["mae", "rmse"],  # mae first
            lower_is_better=True,
        )

        mock_evaluations = [MagicMock(), MagicMock()]
        preferred_index = decision_maker.decide(mock_evaluations)

        # Should use mae, second model has lower mae
        assert preferred_index == 1


class TestVisualDecisionMaker:
    def test_instantiation(self):
        """Test that VisualDecisionMaker can be instantiated."""
        decision_maker = VisualDecisionMaker()
        assert isinstance(decision_maker, DecisionMaker)
