"""Tests for DecisionMaker implementations."""

from unittest.mock import MagicMock, patch

import pytest

from chap_core.preference_learning.decision_maker import (
    MetricBasedDecisionMaker,
)
from chap_core.preference_learning.preference_learner import ModelCandidate


class TestMetricBasedDecisionMaker:
    @pytest.fixture
    def mock_evaluation(self):
        """Create a mock evaluation with configurable metrics."""

        def _create_mock(metrics: dict):
            mock = MagicMock()
            mock_flat_data = MagicMock()
            mock.to_flat.return_value = mock_flat_data
            return mock, metrics

        return _create_mock

    def test_lower_is_better(self, mock_evaluation):
        """Test that lower values are preferred when lower_is_better=True."""
        decision_maker = MetricBasedDecisionMaker(
            metric_name="mae",
            lower_is_better=True,
        )

        model_a = ModelCandidate(model_name="model_a")
        model_b = ModelCandidate(model_name="model_b")

        eval_a, metrics_a = mock_evaluation({"mae": 0.8, "rmse": 1.0})
        eval_b, metrics_b = mock_evaluation({"mae": 0.5, "rmse": 0.7})

        with patch.object(decision_maker, "_compute_metrics") as mock_compute:
            mock_compute.side_effect = [metrics_a, metrics_b]

            preferred, result_metrics_a, result_metrics_b = decision_maker.decide(model_a, eval_a, model_b, eval_b)

        assert preferred == model_b
        assert result_metrics_a["mae"] == 0.8
        assert result_metrics_b["mae"] == 0.5

    def test_higher_is_better(self, mock_evaluation):
        """Test that higher values are preferred when lower_is_better=False."""
        decision_maker = MetricBasedDecisionMaker(
            metric_name="accuracy",
            lower_is_better=False,
        )

        model_a = ModelCandidate(model_name="model_a")
        model_b = ModelCandidate(model_name="model_b")

        eval_a, metrics_a = mock_evaluation({"accuracy": 0.9})
        eval_b, metrics_b = mock_evaluation({"accuracy": 0.7})

        with patch.object(decision_maker, "_compute_metrics") as mock_compute:
            mock_compute.side_effect = [metrics_a, metrics_b]

            preferred, _, _ = decision_maker.decide(model_a, eval_a, model_b, eval_b)

        assert preferred == model_a

    def test_equal_values_prefers_first(self, mock_evaluation):
        """Test that equal metric values prefer model_a."""
        decision_maker = MetricBasedDecisionMaker(
            metric_name="mae",
            lower_is_better=True,
        )

        model_a = ModelCandidate(model_name="model_a")
        model_b = ModelCandidate(model_name="model_b")

        eval_a, metrics_a = mock_evaluation({"mae": 0.5})
        eval_b, metrics_b = mock_evaluation({"mae": 0.5})

        with patch.object(decision_maker, "_compute_metrics") as mock_compute:
            mock_compute.side_effect = [metrics_a, metrics_b]

            preferred, _, _ = decision_maker.decide(model_a, eval_a, model_b, eval_b)

        assert preferred == model_a
