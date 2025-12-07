"""
PreferenceLearner for discovering user preferences through A/B testing.

This module implements a preference learning algorithm that proposes model pairs,
collects feedback on which model is preferred, and learns to propose better models.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelCandidate(BaseModel):
    """A model candidate with its configuration."""

    model_name: str
    configuration: dict = {}

    def __hash__(self):
        return hash((self.model_name, json.dumps(self.configuration, sort_keys=True)))

    def __eq__(self, other):
        if not isinstance(other, ModelCandidate):
            return False
        return self.model_name == other.model_name and self.configuration == other.configuration


@dataclass
class ComparisonResult:
    """Result of comparing two models."""

    model_a: ModelCandidate
    model_b: ModelCandidate
    preferred: ModelCandidate
    metrics_a: dict
    metrics_b: dict
    iteration: int


@dataclass
class PreferenceLearnerState:
    """Serializable state for PreferenceLearner."""

    candidates: list[ModelCandidate] = field(default_factory=list)
    comparison_history: list[ComparisonResult] = field(default_factory=list)
    current_iteration: int = 0
    best_candidate: Optional[ModelCandidate] = None

    def to_dict(self) -> dict:
        """Convert state to dictionary for JSON serialization."""
        return {
            "candidates": [c.model_dump() for c in self.candidates],
            "comparison_history": [
                {
                    "model_a": r.model_a.model_dump(),
                    "model_b": r.model_b.model_dump(),
                    "preferred": r.preferred.model_dump(),
                    "metrics_a": r.metrics_a,
                    "metrics_b": r.metrics_b,
                    "iteration": r.iteration,
                }
                for r in self.comparison_history
            ],
            "current_iteration": self.current_iteration,
            "best_candidate": self.best_candidate.model_dump() if self.best_candidate else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceLearnerState":
        """Create state from dictionary."""
        return cls(
            candidates=[ModelCandidate(**c) for c in data.get("candidates", [])],
            comparison_history=[
                ComparisonResult(
                    model_a=ModelCandidate(**r["model_a"]),
                    model_b=ModelCandidate(**r["model_b"]),
                    preferred=ModelCandidate(**r["preferred"]),
                    metrics_a=r["metrics_a"],
                    metrics_b=r["metrics_b"],
                    iteration=r["iteration"],
                )
                for r in data.get("comparison_history", [])
            ],
            current_iteration=data.get("current_iteration", 0),
            best_candidate=ModelCandidate(**data["best_candidate"]) if data.get("best_candidate") else None,
        )


class PreferenceLearner:
    """
    Learn user preferences for models through iterative A/B testing.

    The learner proposes pairs of models, collects feedback on which is preferred,
    and uses this information to propose better model candidates over time.
    """

    def __init__(
        self,
        candidates: list[ModelCandidate],
        state_file: Optional[Path] = None,
        max_iterations: int = 10,
    ):
        """
        Initialize PreferenceLearner.

        Args:
            candidates: List of model candidates to explore
            state_file: Optional path to file for persisting state
            max_iterations: Maximum number of comparison iterations
        """
        self._state_file = state_file
        self._max_iterations = max_iterations

        # Load state from file if it exists, otherwise initialize
        if state_file and state_file.exists():
            self._state = self._load_state()
            logger.info(f"Loaded state from {state_file}, iteration {self._state.current_iteration}")
        else:
            self._state = PreferenceLearnerState(candidates=candidates)
            logger.info(f"Initialized new PreferenceLearner with {len(candidates)} candidates")

    def _load_state(self) -> PreferenceLearnerState:
        """Load state from file."""
        with open(self._state_file, "r") as f:
            data = json.load(f)
        return PreferenceLearnerState.from_dict(data)

    def _save_state(self) -> None:
        """Save state to file."""
        if self._state_file:
            with open(self._state_file, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
            logger.info(f"Saved state to {self._state_file}")

    def get_next_pair(self) -> Optional[tuple[ModelCandidate, ModelCandidate]]:
        """
        Get the next pair of models to compare.

        Returns:
            Tuple of two ModelCandidates, or None if learning is complete.
        """
        if self._state.current_iteration >= self._max_iterations:
            logger.info("Max iterations reached")
            return None

        if len(self._state.candidates) < 2:
            logger.info("Not enough candidates remaining")
            return None

        # Simple strategy: compare candidates that haven't been compared yet
        # or that have won previous comparisons
        candidates = self._get_candidates_for_comparison()
        if len(candidates) < 2:
            logger.info("No more pairs to compare")
            return None

        return (candidates[0], candidates[1])

    def _get_candidates_for_comparison(self) -> list[ModelCandidate]:
        """
        Select candidates for the next comparison.

        Uses a simple tournament-style approach: winners advance.
        """
        if not self._state.comparison_history:
            # First comparison: use first two candidates
            return self._state.candidates[:2]

        # Get winners from previous comparisons
        winners = {r.preferred for r in self._state.comparison_history}

        # Prioritize winners, then add uncompared candidates
        compared = set()
        for r in self._state.comparison_history:
            compared.add(r.model_a)
            compared.add(r.model_b)

        uncompared = [c for c in self._state.candidates if c not in compared]

        # Build candidate list: winners first, then uncompared
        result = list(winners) + uncompared

        # If we only have winners left, re-compare them
        if len(result) < 2 and len(winners) >= 2:
            result = list(winners)

        return result[:2] if len(result) >= 2 else result

    def report_preference(
        self,
        model_a: ModelCandidate,
        model_b: ModelCandidate,
        preferred: ModelCandidate,
        metrics_a: dict,
        metrics_b: dict,
    ) -> None:
        """
        Report the result of a model comparison.

        Args:
            model_a: First model in comparison
            model_b: Second model in comparison
            preferred: The model that was preferred
            metrics_a: Metrics computed for model_a
            metrics_b: Metrics computed for model_b
        """
        result = ComparisonResult(
            model_a=model_a,
            model_b=model_b,
            preferred=preferred,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            iteration=self._state.current_iteration,
        )
        self._state.comparison_history.append(result)
        self._state.current_iteration += 1
        self._state.best_candidate = preferred

        logger.info(f"Iteration {result.iteration}: {preferred.model_name} preferred")
        logger.info(f"Metrics A: {metrics_a}")
        logger.info(f"Metrics B: {metrics_b}")

        self._save_state()

    def is_complete(self) -> bool:
        """Check if learning is complete."""
        return self._state.current_iteration >= self._max_iterations or self.get_next_pair() is None

    def get_best_candidate(self) -> Optional[ModelCandidate]:
        """Get the current best candidate based on comparison history."""
        return self._state.best_candidate

    def get_comparison_history(self) -> list[ComparisonResult]:
        """Get the full comparison history."""
        return self._state.comparison_history

    @property
    def current_iteration(self) -> int:
        """Get the current iteration number."""
        return self._state.current_iteration
