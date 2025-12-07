"""Tests for DecisionMaker implementations."""

from unittest.mock import MagicMock, patch

import pytest

from chap_core.preference_learning.decision_maker import (
    DecisionMaker,
    MetricDecisionMaker,
    VisualDecisionMaker,
)
from chap_core.preference_learning.preference_learner import ModelCandidate


class TestDecisionMakerInterface:
    def test_decide_returns_int(self):
        """Test that decide returns an integer index."""
        metrics = [{"mae": 0.8}, {"mae": 0.5}]
        decision_maker = MetricDecisionMaker(
            metrics=metrics,
            metric_name="mae",
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
            metric_name="mae",
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
            metric_name="accuracy",
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
            metric_name="mae",
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
            metric_name="mae",
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
            metric_name="mae",
            lower_is_better=True,
        )

        mock_evaluations = [MagicMock(), MagicMock()]  # 2 evaluations

        with pytest.raises(ValueError, match="Expected 2 metrics"):
            decision_maker.decide(mock_evaluations)


class TestVisualDecisionMaker:
    def test_model_names_stored(self):
        """Test that model names are stored for display."""
        model_names = ["model_a", "model_b"]
        decision_maker = VisualDecisionMaker(model_names=model_names)
        assert decision_maker._model_names == model_names
