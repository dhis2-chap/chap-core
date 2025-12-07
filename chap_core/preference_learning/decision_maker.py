"""
DecisionMaker for selecting preferred models from evaluation results.

This module provides interfaces and implementations for deciding which model
is preferred given two evaluation results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from chap_core.assessment.evaluation import Evaluation
from chap_core.preference_learning.preference_learner import ModelCandidate

logger = logging.getLogger(__name__)


class DecisionMaker(ABC):
    """
    Abstract base class for making decisions between two model evaluations.

    Implementations can use different strategies: metric-based, user input,
    or more sophisticated preference models.
    """

    @abstractmethod
    def decide(
        self,
        model_a: ModelCandidate,
        evaluation_a: Evaluation,
        model_b: ModelCandidate,
        evaluation_b: Evaluation,
    ) -> tuple[ModelCandidate, dict, dict]:
        """
        Decide which model is preferred.

        Args:
            model_a: First model candidate
            evaluation_a: Evaluation results for model_a
            model_b: Second model candidate
            evaluation_b: Evaluation results for model_b

        Returns:
            Tuple of (preferred model, metrics for model_a, metrics for model_b)
        """
        pass


class MetricBasedDecisionMaker(DecisionMaker):
    """
    Decision maker that selects based on a specific metric.

    Computes metrics from evaluations and selects the model with the
    better metric value.
    """

    def __init__(
        self,
        metric_name: str = "mae",
        lower_is_better: bool = True,
    ):
        """
        Initialize MetricBasedDecisionMaker.

        Args:
            metric_name: Name of metric to use for comparison
            lower_is_better: If True, lower metric values are preferred
        """
        self._metric_name = metric_name
        self._lower_is_better = lower_is_better

    def _compute_metrics(self, evaluation: Evaluation) -> dict:
        """
        Compute metrics from an evaluation.

        Args:
            evaluation: Evaluation object with forecasts and observations

        Returns:
            Dictionary of metric name to value
        """
        from chap_core.rest_api.v1.routers.analytics import calculate_all_metrics

        flat_data = evaluation.to_flat()
        metrics = calculate_all_metrics(flat_data.forecasts, flat_data.observations)
        return metrics

    def decide(
        self,
        model_a: ModelCandidate,
        evaluation_a: Evaluation,
        model_b: ModelCandidate,
        evaluation_b: Evaluation,
    ) -> tuple[ModelCandidate, dict, dict]:
        """
        Decide based on comparing a specific metric.

        Args:
            model_a: First model candidate
            evaluation_a: Evaluation results for model_a
            model_b: Second model candidate
            evaluation_b: Evaluation results for model_b

        Returns:
            Tuple of (preferred model, metrics for model_a, metrics for model_b)
        """
        metrics_a = self._compute_metrics(evaluation_a)
        metrics_b = self._compute_metrics(evaluation_b)

        value_a = metrics_a.get(self._metric_name)
        value_b = metrics_b.get(self._metric_name)

        if value_a is None or value_b is None:
            logger.warning(f"Metric {self._metric_name} not found, using first available metric")
            # Fall back to first available metric
            for key in metrics_a:
                if key in metrics_b:
                    self._metric_name = key
                    value_a = metrics_a[key]
                    value_b = metrics_b[key]
                    break

        logger.info(f"Model A ({model_a.model_name}): {self._metric_name}={value_a}")
        logger.info(f"Model B ({model_b.model_name}): {self._metric_name}={value_b}")

        if self._lower_is_better:
            preferred = model_a if value_a <= value_b else model_b
        else:
            preferred = model_a if value_a >= value_b else model_b

        logger.info(f"Preferred: {preferred.model_name} (based on {self._metric_name})")

        return preferred, metrics_a, metrics_b


class InteractiveDecisionMaker(DecisionMaker):
    """
    Decision maker that prompts the user to select their preferred model.

    Displays metrics for both models and asks for user input.
    """

    def __init__(self, display_metrics: Optional[list[str]] = None):
        """
        Initialize InteractiveDecisionMaker.

        Args:
            display_metrics: List of metric names to display (None for all)
        """
        self._display_metrics = display_metrics

    def _compute_metrics(self, evaluation: Evaluation) -> dict:
        """Compute metrics from an evaluation."""
        from chap_core.rest_api.v1.routers.analytics import calculate_all_metrics

        flat_data = evaluation.to_flat()
        metrics = calculate_all_metrics(flat_data.forecasts, flat_data.observations)
        return metrics

    def _display_comparison(
        self,
        model_a: ModelCandidate,
        metrics_a: dict,
        model_b: ModelCandidate,
        metrics_b: dict,
    ) -> None:
        """Display metrics comparison to user."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        metrics_to_show = self._display_metrics or list(metrics_a.keys())

        print(f"\n{'Metric':<20} {'Model A':<20} {'Model B':<20}")
        print("-" * 60)
        print(f"{'Name':<20} {model_a.model_name:<20} {model_b.model_name:<20}")
        print("-" * 60)

        for metric in metrics_to_show:
            val_a = metrics_a.get(metric, "N/A")
            val_b = metrics_b.get(metric, "N/A")
            if isinstance(val_a, float):
                val_a = f"{val_a:.4f}"
            if isinstance(val_b, float):
                val_b = f"{val_b:.4f}"
            print(f"{metric:<20} {str(val_a):<20} {str(val_b):<20}")

        print("=" * 60)

    def decide(
        self,
        model_a: ModelCandidate,
        evaluation_a: Evaluation,
        model_b: ModelCandidate,
        evaluation_b: Evaluation,
    ) -> tuple[ModelCandidate, dict, dict]:
        """
        Ask user to decide which model is preferred.

        Args:
            model_a: First model candidate
            evaluation_a: Evaluation results for model_a
            model_b: Second model candidate
            evaluation_b: Evaluation results for model_b

        Returns:
            Tuple of (preferred model, metrics for model_a, metrics for model_b)
        """
        metrics_a = self._compute_metrics(evaluation_a)
        metrics_b = self._compute_metrics(evaluation_b)

        self._display_comparison(model_a, metrics_a, model_b, metrics_b)

        while True:
            choice = input("\nWhich model do you prefer? [A/B]: ").strip().upper()
            if choice == "A":
                preferred = model_a
                break
            elif choice == "B":
                preferred = model_b
                break
            else:
                print("Please enter 'A' or 'B'")

        return preferred, metrics_a, metrics_b
