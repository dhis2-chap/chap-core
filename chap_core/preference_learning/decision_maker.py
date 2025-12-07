"""
DecisionMaker for selecting preferred models from evaluation results.

This module provides interfaces and implementations for deciding which model
is preferred given a list of evaluation results.
"""

import logging
import tempfile
import webbrowser
from abc import ABC, abstractmethod

from chap_core.assessment.evaluation import Evaluation

logger = logging.getLogger(__name__)


class DecisionMaker(ABC):
    """
    Abstract base class for making decisions between model evaluations.

    The DecisionMaker receives a list of Evaluations and returns the index
    of the preferred one. It does not compute metrics - that responsibility
    belongs to the caller.
    """

    @abstractmethod
    def decide(self, evaluations: list[Evaluation]) -> int:
        """
        Decide which evaluation is preferred.

        Args:
            evaluations: List of Evaluation objects to compare

        Returns:
            Index of the preferred evaluation (0-based)
        """
        pass


class VisualDecisionMaker(DecisionMaker):
    """
    Decision maker that displays backtest plots and asks user to choose.

    Shows visual comparison of model predictions vs observations and
    prompts the user to select their preferred model.
    """

    def __init__(self, model_names: list[str] | None = None):
        """
        Initialize VisualDecisionMaker.

        Args:
            model_names: Optional list of model names for display purposes
        """
        self._model_names = model_names

    def decide(self, evaluations: list[Evaluation]) -> int:
        """
        Display backtest plots and ask user to choose preferred model.

        Args:
            evaluations: List of Evaluation objects to compare

        Returns:
            Index of the preferred evaluation
        """
        from chap_core.plotting.backtest_plot import EvaluationBackTestPlot

        # Generate and display plots for each evaluation
        for i, evaluation in enumerate(evaluations):
            model_name = self._model_names[i] if self._model_names and i < len(self._model_names) else f"Model {i + 1}"

            logger.info(f"Generating backtest plot for {model_name}")

            backtest = evaluation.to_backtest()
            plot = EvaluationBackTestPlot.from_backtest(backtest)
            chart = plot.plot()

            # Save to temp file and open in browser
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                chart.save(f.name)
                logger.info(f"Opening plot for {model_name}: {f.name}")
                webbrowser.open(f"file://{f.name}")

        # Prompt user for choice
        print("\n" + "=" * 60)
        print("MODEL COMPARISON - Please review the plots")
        print("=" * 60)

        for i in range(len(evaluations)):
            model_name = self._model_names[i] if self._model_names and i < len(self._model_names) else f"Model {i + 1}"
            print(f"  [{i + 1}] {model_name}")

        print("=" * 60)

        while True:
            try:
                choice = input(f"\nWhich model do you prefer? [1-{len(evaluations)}]: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(evaluations):
                    return idx
                print(f"Please enter a number between 1 and {len(evaluations)}")
            except ValueError:
                print(f"Please enter a number between 1 and {len(evaluations)}")


class MetricDecisionMaker(DecisionMaker):
    """
    Decision maker that selects based on pre-computed metrics.

    This is a simple automatic decision maker that compares a specific
    metric value across evaluations.
    """

    def __init__(
        self,
        metrics: list[dict],
        metric_name: str = "mae",
        lower_is_better: bool = True,
    ):
        """
        Initialize MetricDecisionMaker.

        Args:
            metrics: List of metric dictionaries, one per evaluation
            metric_name: Name of metric to use for comparison
            lower_is_better: If True, lower metric values are preferred
        """
        self._metrics = metrics
        self._metric_name = metric_name
        self._lower_is_better = lower_is_better

    def decide(self, evaluations: list[Evaluation]) -> int:
        """
        Decide based on comparing a specific metric.

        Args:
            evaluations: List of Evaluation objects (used for count validation)

        Returns:
            Index of the preferred evaluation
        """
        if len(self._metrics) != len(evaluations):
            raise ValueError(f"Expected {len(evaluations)} metrics, got {len(self._metrics)}")

        values = [m.get(self._metric_name) for m in self._metrics]

        # Handle missing metric
        if any(v is None for v in values):
            logger.warning(f"Metric {self._metric_name} not found in all evaluations")
            # Find first common metric
            common_metrics = set(self._metrics[0].keys())
            for m in self._metrics[1:]:
                common_metrics &= set(m.keys())
            if common_metrics:
                self._metric_name = next(iter(common_metrics))
                values = [m.get(self._metric_name) for m in self._metrics]
                logger.info(f"Using fallback metric: {self._metric_name}")

        for i, v in enumerate(values):
            logger.info(f"Model {i + 1}: {self._metric_name}={v}")

        if self._lower_is_better:
            best_idx = min(range(len(values)), key=lambda idx: values[idx])
        else:
            best_idx = max(range(len(values)), key=lambda idx: values[idx])

        logger.info(f"Preferred: Model {best_idx + 1} (based on {self._metric_name})")

        return best_idx
