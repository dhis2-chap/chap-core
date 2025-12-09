# Preference Learning

This document explains the preference learning system and how to implement a custom `PreferenceLearner` algorithm.

## Overview

Preference learning is a technique for discovering optimal model configurations through iterative A/B testing. Instead of using automated optimization metrics alone, it can incorporate human judgment to select preferred models based on visual inspection of backtest plots.

The system works by:
1. Generating candidate model configurations from a hyperparameter search space
2. Running backtests on pairs of candidates
3. Presenting results to a decision maker (human or automated)
4. Learning from preferences to propose better candidates
5. Repeating until convergence or max iterations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     preference_learn CLI                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load dataset and search space                               │
│                                                                  │
│  2. Initialize PreferenceLearner                                │
│     └── PreferenceLearner.init(model_name, search_space)        │
│                                                                  │
│  3. Main loop:                                                  │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ candidates = learner.get_next_candidates()          │     │
│     │                                                      │     │
│     │ for each candidate:                                  │     │
│     │   └── Run backtest → Evaluation                     │     │
│     │                                                      │     │
│     │ metrics = compute_metrics(evaluations)              │     │
│     │                                                      │     │
│     │ preferred_idx = decision_maker.decide(evaluations)  │     │
│     │                                                      │     │
│     │ learner.report_preference(candidates,               │     │
│     │                           preferred_idx, metrics)   │     │
│     │                                                      │     │
│     │ learner.save(state_file)                            │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                  │
│  4. best = learner.get_best_candidate()                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### ModelCandidate

Represents a model configuration to evaluate:

```python
from chap_core.preference_learning.preference_learner import ModelCandidate

candidate = ModelCandidate(
    model_name="my_model",
    configuration={"learning_rate": 0.01, "hidden_size": 128}
)
```

### ComparisonResult

Records the result of comparing candidates:

```python
from chap_core.preference_learning.preference_learner import ComparisonResult

result = ComparisonResult(
    candidates=[candidate_a, candidate_b],
    preferred_index=1,  # candidate_b was preferred
    metrics=[{"mae": 0.8}, {"mae": 0.5}],
    iteration=0
)

# Access the winner
winner = result.preferred  # Returns candidate_b
```

### Search Space

The search space defines hyperparameters to explore. It uses parsed values from `load_search_space_from_config()`:

```python
from chap_core.hpo.base import load_search_space_from_config, Int, Float

# Raw YAML config format:
raw_config = {
    "learning_rate": {"low": 0.001, "high": 0.1, "type": "float", "log": True},
    "hidden_size": {"low": 32, "high": 256, "type": "int"},
    "optimizer": {"values": ["adam", "sgd", "rmsprop"]}
}

# Parsed search space contains Int, Float, or list objects:
search_space = load_search_space_from_config(raw_config)
# Result: {
#     "learning_rate": Float(low=0.001, high=0.1, log=True),
#     "hidden_size": Int(low=32, high=256),
#     "optimizer": ["adam", "sgd", "rmsprop"]
# }
```

## Implementing a PreferenceLearner

To implement a custom preference learning algorithm, subclass `PreferenceLearnerBase`:

```python
from pathlib import Path
from typing import Any, Optional
from chap_core.preference_learning.preference_learner import (
    PreferenceLearnerBase,
    ModelCandidate,
    ComparisonResult,
    SearchSpaceValue,
)


class MyPreferenceLearner(PreferenceLearnerBase):
    """
    Custom preference learning algorithm.

    This could implement:
    - Bayesian optimization with preference feedback
    - Multi-armed bandit approaches
    - Genetic algorithms
    - etc.
    """

    def __init__(self, state: MyLearnerState):
        """Initialize with internal state."""
        self._state = state

    @classmethod
    def init(
        cls,
        model_name: str,
        search_space: dict[str, SearchSpaceValue],
        max_iterations: int = 10,
    ) -> "MyPreferenceLearner":
        """
        Initialize a new learner with a parsed search space.

        Args:
            model_name: Name of the model template to optimize
            search_space: Parsed search space dict mapping param names to
                         Int, Float, or list of categorical values.
            max_iterations: Maximum number of comparison iterations

        Returns:
            New MyPreferenceLearner instance
        """
        # Generate initial candidates from search space
        candidates = cls._generate_candidates(model_name, search_space)

        state = MyLearnerState(
            model_name=model_name,
            candidates=candidates,
            max_iterations=max_iterations,
        )
        return cls(state)

    def save(self, filepath: Path) -> None:
        """
        Save learner state to a file.

        This enables resuming learning across sessions.
        """
        import json
        with open(filepath, "w") as f:
            json.dump(self._state.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "MyPreferenceLearner":
        """Load learner from a saved state file."""
        import json
        with open(filepath, "r") as f:
            data = json.load(f)
        state = MyLearnerState.from_dict(data)
        return cls(state)

    def get_next_candidates(self) -> Optional[list[ModelCandidate]]:
        """
        Get the next set of candidates to compare.

        Returns:
            List of 2+ ModelCandidates to compare, or None if done.

        This is where your algorithm's intelligence lives:
        - Which candidates should be compared next?
        - How do you balance exploration vs exploitation?
        - How do you use past preferences to guide selection?
        """
        if self.is_complete():
            return None

        # Your algorithm logic here
        # Example: select candidates based on uncertainty, expected improvement, etc.
        return [self._state.candidates[0], self._state.candidates[1]]

    def report_preference(
        self,
        candidates: list[ModelCandidate],
        preferred_index: int,
        metrics: list[dict],
    ) -> None:
        """
        Report the result of a comparison.

        Args:
            candidates: The candidates that were compared
            preferred_index: Index of the preferred candidate (0-based)
            metrics: Computed metrics for each candidate

        Use this feedback to update your internal model:
        - Update belief distributions
        - Eliminate inferior candidates
        - Adjust exploration strategy
        """
        result = ComparisonResult(
            candidates=candidates,
            preferred_index=preferred_index,
            metrics=metrics,
            iteration=self._state.current_iteration,
        )
        self._state.comparison_history.append(result)
        self._state.current_iteration += 1
        self._state.best_candidate = candidates[preferred_index]

    def is_complete(self) -> bool:
        """Check if learning is complete."""
        return self._state.current_iteration >= self._state.max_iterations

    def get_best_candidate(self) -> Optional[ModelCandidate]:
        """Get the current best candidate."""
        return self._state.best_candidate

    def get_comparison_history(self) -> list[ComparisonResult]:
        """Get all comparison results."""
        return self._state.comparison_history

    @property
    def current_iteration(self) -> int:
        """Get current iteration number."""
        return self._state.current_iteration
```

## PreferenceLearnerBase Interface

The abstract base class defines these required methods:

| Method | Description |
|--------|-------------|
| `init(model_name, search_space, max_iterations)` | Class method to create a new learner |
| `save(filepath)` | Save state to file for persistence |
| `load(filepath)` | Class method to restore from saved state |
| `get_next_candidates()` | Return next candidates to compare, or None if done |
| `report_preference(candidates, preferred_index, metrics)` | Record comparison result |
| `is_complete()` | Check if learning should stop |
| `get_best_candidate()` | Return current best candidate |
| `get_comparison_history()` | Return all comparison results |
| `current_iteration` | Property returning current iteration number |

## Existing Implementation: TournamentPreferenceLearner

The default implementation uses a tournament-style bracket:

1. **Initialization**: Generates all candidate configurations from the search space
2. **Selection**: Winners advance to compete against other winners or uncompared candidates
3. **Termination**: Stops after max_iterations or when no more pairs to compare

This is a simple but effective baseline. More sophisticated algorithms could:
- Use Bayesian optimization to model the preference function
- Implement Thompson sampling for exploration
- Use Elo ratings to rank candidates
- Apply genetic algorithms with preference-based selection

## DecisionMaker

The `DecisionMaker` determines which candidate is preferred:

```python
from chap_core.preference_learning.decision_maker import (
    DecisionMaker,
    VisualDecisionMaker,
    MetricDecisionMaker,
)

# Visual: Opens plots in browser, user chooses
visual_dm = VisualDecisionMaker()

# Metric: Automatic selection based on computed metrics
metric_dm = MetricDecisionMaker(
    metrics=[{"mae": 0.8}, {"mae": 0.5}],
    metric_names=["mae", "rmse"],  # Priority order
    lower_is_better=True
)

# Use it
preferred_idx = decision_maker.decide(evaluations)
```

## CLI Usage

```bash
chap preference-learn \
    --model-name ../my_model \
    --dataset-csv ./data.csv \
    --search-space-yaml ./search_space.yaml \
    --state-file ./preference_state.json \
    --decision-mode metric \
    --decision-metrics mae rmse \
    --max-iterations 10
```

## Testing Your Implementation

```python
import pytest
from pathlib import Path


class TestMyPreferenceLearner:
    def test_init_creates_candidates(self):
        search_space = {"param": [1, 2, 3]}
        learner = MyPreferenceLearner.init(
            model_name="test",
            search_space=search_space,
            max_iterations=5,
        )
        assert learner.current_iteration == 0
        assert not learner.is_complete()

    def test_get_next_candidates_returns_pair(self):
        learner = MyPreferenceLearner.init(
            model_name="test",
            search_space={"x": [1, 2]},
        )
        candidates = learner.get_next_candidates()
        assert candidates is not None
        assert len(candidates) == 2

    def test_report_preference_updates_state(self):
        learner = MyPreferenceLearner.init(
            model_name="test",
            search_space={"x": [1, 2]},
        )
        candidates = learner.get_next_candidates()
        learner.report_preference(
            candidates=candidates,
            preferred_index=0,
            metrics=[{"mae": 0.5}, {"mae": 0.7}],
        )
        assert learner.current_iteration == 1
        assert learner.get_best_candidate() == candidates[0]

    def test_save_and_load_roundtrip(self, tmp_path):
        learner = MyPreferenceLearner.init(
            model_name="test",
            search_space={"x": [1, 2]},
        )
        candidates = learner.get_next_candidates()
        learner.report_preference(candidates, 0, [{"mae": 0.5}, {"mae": 0.7}])

        state_file = tmp_path / "state.json"
        learner.save(state_file)

        loaded = MyPreferenceLearner.load(state_file)
        assert loaded.current_iteration == 1
        assert len(loaded.get_comparison_history()) == 1
```

## Example Search Space

See `example_data/preference_learning/ewars_hpo_search_space.yaml` for a complete example:

```yaml
# Number of lags to include in the model (integer parameter)
n_lags:
  low: 1
  high: 6
  type: int

# Prior on precision of fixed effects - acts as regularization
# Using log scale since this spans multiple orders of magnitude
precision:
  low: 0.001
  high: 1.0
  type: float
  log: true
```

## File Locations

- `chap_core/preference_learning/preference_learner.py` - Base class and TournamentPreferenceLearner
- `chap_core/preference_learning/decision_maker.py` - DecisionMaker implementations
- `chap_core/cli_endpoints/preference_learn.py` - CLI endpoint
- `tests/preference_learning/` - Tests
- `example_data/preference_learning/` - Example search space files
