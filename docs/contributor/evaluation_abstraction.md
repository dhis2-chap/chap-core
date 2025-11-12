# Evaluation Abstraction Design

## Overview

This document describes the proposed refactoring to create a database-agnostic Evaluation abstraction. The goal is to unify how evaluations/backtests are represented throughout the codebase, enabling better code reuse between the REST API and CLI evaluation workflows.

## Problem Statement

Currently, the codebase has two different approaches to handling model evaluations:

1. **REST API**: Uses the `BackTest` database model (tied to SQLModel/database)
2. **CLI**: Uses GluonTS Evaluator directly without database persistence

This duplication leads to:
- Code that cannot be easily shared between REST API and CLI
- Different evaluation workflows that are hard to maintain
- Tight coupling between evaluation logic and database schema

## Current State Analysis

### Database Model Structure

The current BackTest implementation is defined in `chap_core/database/tables.py:39-47`:

```python
class BackTest(_BackTestRead, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset: DataSet = Relationship()
    forecasts: List["BackTestForecast"] = Relationship(back_populates="backtest", cascade_delete=True)
    metrics: List["BackTestMetric"] = Relationship(back_populates="backtest", cascade_delete=True)
    aggregate_metrics: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    model_db_id: int = Field(foreign_key="configuredmodeldb.id")
    configured_model: Optional["ConfiguredModelDB"] = Relationship()
```

**Key components:**

1. **BackTestForecast** (`tables.py:113-118`): Stores individual forecast predictions
   - Fields: `period`, `org_unit`, `values` (samples), `last_train_period`, `last_seen_period`
   - One record per location-period-split combination

2. **BackTestMetric** (`tables.py:121-137`): Deprecated, not used in new metric system

3. **Related metadata**:
   - `org_units: List[str]` - evaluated locations
   - `split_periods: List[PeriodID]` - train/test split points
   - `model_db_id` - reference to configured model
   - `dataset` - relationship to DataSet table

### REST API Workflow

Location: `chap_core/rest_api/v1/routers/analytics.py` and `chap_core/rest_api/db_worker_functions.py`

**Evaluation Creation Process:**

```
1. POST /create-backtest
   └─> Queue worker: run_backtest()
       └─> Load dataset and configured model
       └─> Call _backtest() -> returns Iterable[DataSet[SamplesWithTruth]]
       └─> session.add_evaluation_results() -> persists to BackTest table
       └─> Returns backtest.id
```

**Data Consumption:**

1. **GET /evaluation-entry** (`analytics.py:217-284`):
   - Queries BackTestForecast records
   - Returns quantiles for specified split_period and org_units
   - Can aggregate to "adm0" level

2. **Metric Computation** (`assessment/metrics/__init__.py:84-102`):
   ```python
   def compute_all_aggregated_metrics_from_backtest(backtest: BackTest):
       flat_forecasts = convert_backtest_to_flat_forecasts(backtest.forecasts)
       flat_observations = convert_backtest_observations_to_flat_observations(
           backtest.dataset.observations
       )
       # Compute metrics using flat representations
       for metric in metrics:
           result = metric.get_metric(flat_observations, flat_forecasts)
   ```

3. **Visualization** (`plotting/evaluation_plot.py:236-243`):
   ```python
   def make_plot_from_backtest_object(backtest: BackTest, plotting_class, metric):
       flat_forecasts = convert_backtest_to_flat_forecasts(backtest.forecasts)
       flat_observations = convert_backtest_observations_to_flat_observations(
           backtest.dataset.observations
       )
       metric_data = metric.compute(flat_observations, flat_forecasts)
       return plotting_class(metric_data).plot_spec()
   ```

**Key Pattern**: BackTest DB object → Flat DataFrame representation → Metrics/Visualization

### CLI Workflow

Location: `chap_core/cli.py:189-309` and `chap_core/assessment/prediction_evaluator.py:58-118`

**Evaluation Process:**

```
1. cli.py evaluate command
   └─> Load model template and get configured model
   └─> Call evaluate_model(estimator, data, ...)
       └─> Uses train_test_generator() for data splits
       └─> estimator.train() and predictor.predict()
       └─> Uses GluonTS Evaluator directly
       └─> Returns (aggregate_metrics, item_metrics) tuple
   └─> Save results to CSV files
   └─> No database persistence
```

**Key Differences from REST API:**
- No BackTest database model used
- Results stay in memory as Python dicts/tuples
- Direct use of GluonTS evaluation
- CSV export instead of database storage

### Flat Representation System

Location: `chap_core/assessment/flat_representations.py`

The codebase has a well-established flat representation system for working with evaluation data:

**FlatForecasts**: Tabular format for forecasts
```
Columns: location, time_period, horizon_distance, sample, forecast
Example:
location  | time_period | horizon_distance | sample | forecast
----------|-------------|------------------|--------|----------
region_A  | 2024-01     | 1                | 0      | 45.2
region_A  | 2024-01     | 1                | 1      | 48.7
region_A  | 2024-02     | 2                | 0      | 52.1
...
```

**FlatObserved**: Tabular format for observations
```
Columns: location, time_period, disease_cases
Example:
location  | time_period | disease_cases
----------|-------------|---------------
region_A  | 2024-01     | 47.0
region_A  | 2024-02     | 51.5
...
```

**Conversion Functions:**

1. `convert_backtest_to_flat_forecasts(backtest_forecasts: List[BackTestForecast])`:
   - Converts BackTestForecast records to FlatForecasts DataFrame
   - Calculates `horizon_distance` from period differences
   - Unpacks sample arrays into individual rows

2. `convert_backtest_observations_to_flat_observations(observations: List[ObservationBase])`:
   - Extracts disease_cases observations
   - Returns FlatObserved DataFrame

**Usage**: All metrics and visualization code works with flat representations, not database models directly.

### SamplesWithTruth Intermediate Format

Location: `chap_core/datatypes.py:361`

During evaluation, results are generated as `DataSet[SamplesWithTruth]`:

```python
@tsdataclass
class SamplesWithTruth(Samples):
    disease_cases: float  # truth value
    # Inherited from Samples:
    # time_period: TimePeriod
    # samples: np.ndarray  # forecast samples
```

This is the in-memory format returned by `_backtest()` and then persisted to database via `add_evaluation_results()`.

### Current Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         REST API Path                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  _backtest()                                                  │
│      ↓                                                        │
│  Iterable[DataSet[SamplesWithTruth]]                         │
│      ↓                                                        │
│  session.add_evaluation_results()                            │
│      ↓                                                        │
│  BackTest (DB) ← ── ── stored in database                    │
│      ├─> BackTestForecast records                            │
│      └─> DataSet relationship                                │
│      ↓                                                        │
│  convert_backtest_to_flat_*()                                │
│      ↓                                                        │
│  FlatForecasts + FlatObserved DataFrames                     │
│      ↓                                                        │
│  Metrics / Visualization                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                          CLI Path                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  evaluate_model()                                             │
│      ↓                                                        │
│  GluonTS Evaluator                                            │
│      ↓                                                        │
│  (aggregate_metrics, item_metrics) tuples                    │
│      ↓                                                        │
│  Save to CSV                                                  │
│      ↓                                                        │
│  No database persistence                                     │
└─────────────────────────────────────────────────────────────┘
```

## Proposed Design

### Core Concept: EvaluationBase ABC

Create an abstract base class that defines the interface for all evaluation representations, decoupled from database implementation:

```python
from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class EvaluationBase(ABC):
    """
    Abstract base class for evaluation results.

    An Evaluation represents the complete results of evaluating a model:
    - Forecasts (with samples/quantiles)
    - Observations (ground truth)
    - Metadata (locations, split periods)

    This abstraction is database-agnostic and can be implemented by
    different concrete classes (database-backed, in-memory, etc.).
    """

    @abstractmethod
    def to_flat_forecasts(self) -> pd.DataFrame:
        """
        Export forecasts in FlatForecasts format.

        Returns:
            DataFrame with columns: location, time_period,
                                   horizon_distance, sample, forecast
        """
        pass

    @abstractmethod
    def to_flat_observations(self) -> pd.DataFrame:
        """
        Export observations in FlatObserved format.

        Returns:
            DataFrame with columns: location, time_period, disease_cases
        """
        pass

    @abstractmethod
    def get_org_units(self) -> List[str]:
        """
        Get list of locations included in this evaluation.

        Returns:
            List of location identifiers (org_units)
        """
        pass

    @abstractmethod
    def get_split_periods(self) -> List[str]:
        """
        Get list of train/test split periods used in evaluation.

        Returns:
            List of period identifiers (e.g., ["2024-01", "2024-02"])
        """
        pass

    def compute_metrics(self, metrics: List[MetricBase]) -> pd.DataFrame:
        """
        Compute metrics on this evaluation.

        Default implementation uses flat representations.
        Subclasses can override for optimization.

        Args:
            metrics: List of metric objects to compute

        Returns:
            DataFrame with computed metric values
        """
        from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved

        forecasts = FlatForecasts(self.to_flat_forecasts())
        observations = FlatObserved(self.to_flat_observations())

        results = []
        for metric in metrics:
            metric_df = metric.get_metric(observations, forecasts)
            results.append(metric_df)

        return pd.concat(results, ignore_index=True)
```

### Concrete Implementation: BacktestEvaluation

Wraps existing BackTest database model to implement the abstract interface:

```python
class BacktestEvaluation(EvaluationBase):
    """
    Evaluation implementation backed by database BackTest model.

    This wraps an existing BackTest object and provides the
    EvaluationBase interface without modifying the database schema.
    """

    def __init__(self, backtest: BackTest):
        """
        Args:
            backtest: Database BackTest object
        """
        self._backtest = backtest
        self._flat_forecasts_cache = None
        self._flat_observations_cache = None

    @classmethod
    def from_backtest(cls, backtest: BackTest) -> "BacktestEvaluation":
        """
        Create BacktestEvaluation from database BackTest object.

        Args:
            backtest: Database BackTest object (with relationships loaded)

        Returns:
            BacktestEvaluation instance
        """
        return cls(backtest)

    def to_backtest(self) -> BackTest:
        """
        Get underlying database BackTest object.

        Returns:
            BackTest database model
        """
        return self._backtest

    def to_flat_forecasts(self) -> pd.DataFrame:
        """Export forecasts using existing conversion function."""
        if self._flat_forecasts_cache is None:
            from chap_core.assessment.flat_representations import (
                convert_backtest_to_flat_forecasts
            )
            self._flat_forecasts_cache = convert_backtest_to_flat_forecasts(
                self._backtest.forecasts
            )
        return self._flat_forecasts_cache

    def to_flat_observations(self) -> pd.DataFrame:
        """Export observations using existing conversion function."""
        if self._flat_observations_cache is None:
            from chap_core.assessment.flat_representations import (
                convert_backtest_observations_to_flat_observations
            )
            self._flat_observations_cache = (
                convert_backtest_observations_to_flat_observations(
                    self._backtest.dataset.observations
                )
            )
        return self._flat_observations_cache

    def get_org_units(self) -> List[str]:
        """Get locations from BackTest metadata."""
        return self._backtest.org_units

    def get_split_periods(self) -> List[str]:
        """Get split periods from BackTest metadata."""
        return self._backtest.split_periods
```

### Future Implementation: InMemoryEvaluation

For CLI and other non-database use cases:

```python
class InMemoryEvaluation(EvaluationBase):
    """
    Evaluation implementation using in-memory data structures.

    Suitable for CLI workflows where database persistence is not needed.
    Can be created directly from evaluation results or flat DataFrames.
    """

    def __init__(
        self,
        forecasts_df: pd.DataFrame,
        observations_df: pd.DataFrame,
        org_units: List[str],
        split_periods: List[str],
    ):
        """
        Args:
            forecasts_df: DataFrame in FlatForecasts format
            observations_df: DataFrame in FlatObserved format
            org_units: List of location identifiers
            split_periods: List of split period identifiers
        """
        self._forecasts = forecasts_df
        self._observations = observations_df
        self._org_units = org_units
        self._split_periods = split_periods

    @classmethod
    def from_samples_with_truth(
        cls,
        results: Iterable[DataSet[SamplesWithTruth]],
        last_train_period: TimePeriod,
    ) -> "InMemoryEvaluation":
        """
        Create from _backtest() results without database persistence.

        Args:
            results: Iterator of DataSet[SamplesWithTruth] from backtest
            last_train_period: Final training period

        Returns:
            InMemoryEvaluation instance
        """
        # Convert SamplesWithTruth to flat representations
        # (implementation details omitted for brevity)
        pass

    def to_flat_forecasts(self) -> pd.DataFrame:
        """Return forecasts DataFrame directly."""
        return self._forecasts

    def to_flat_observations(self) -> pd.DataFrame:
        """Return observations DataFrame directly."""
        return self._observations

    def get_org_units(self) -> List[str]:
        """Return stored org_units."""
        return self._org_units

    def get_split_periods(self) -> List[str]:
        """Return stored split_periods."""
        return self._split_periods

    def to_backtest(self, session: SessionWrapper, info: BackTestCreate) -> BackTest:
        """
        Persist to database as BackTest.

        Args:
            session: Database session wrapper
            info: Metadata for creating BackTest record

        Returns:
            Persisted BackTest object
        """
        # Convert flat representations to BackTest structure
        # (implementation details omitted for brevity)
        pass
```

## API Usage Examples

### Example 1: REST API - Loading and Computing Metrics

```python
# Current approach (tightly coupled to database)
backtest = session.get_backtest(backtest_id)
flat_forecasts = convert_backtest_to_flat_forecasts(backtest.forecasts)
flat_observations = convert_backtest_observations_to_flat_observations(
    backtest.dataset.observations
)
metrics_df = compute_metrics(flat_observations, flat_forecasts)

# Proposed approach (using abstraction)
backtest_db = session.get_backtest(backtest_id)
evaluation = BacktestEvaluation.from_backtest(backtest_db)

# Common interface regardless of implementation
metrics_df = evaluation.compute_metrics([RMSE(), MAE(), CRPS()])
org_units = evaluation.get_org_units()
split_periods = evaluation.get_split_periods()
```

### Example 2: Visualization

```python
# Current approach
def make_plot_from_backtest_object(backtest: BackTest, metric):
    flat_forecasts = convert_backtest_to_flat_forecasts(backtest.forecasts)
    flat_observations = convert_backtest_observations_to_flat_observations(
        backtest.dataset.observations
    )
    metric_data = metric.compute(flat_observations, flat_forecasts)
    return plot(metric_data)

# Proposed approach (works with any EvaluationBase implementation)
def make_plot_from_evaluation(evaluation: EvaluationBase, metric):
    flat_forecasts = evaluation.to_flat_forecasts()
    flat_observations = evaluation.to_flat_observations()
    metric_data = metric.compute(flat_observations, flat_forecasts)
    return plot(metric_data)

# Usage
evaluation = BacktestEvaluation.from_backtest(backtest_db)
chart = make_plot_from_evaluation(evaluation, RMSE())
```

### Example 3: CLI Evaluation (Future)

```python
# Current CLI approach
def evaluate(data, model_name, ...):
    estimator = load_model(model_name)
    aggregate_metrics, item_metrics = evaluate_model(estimator, data)
    save_to_csv(aggregate_metrics, "results.csv")

# Proposed approach with InMemoryEvaluation
def evaluate(data, model_name, ...):
    estimator = load_model(model_name)
    results = _backtest(estimator, data)

    # Create in-memory evaluation (no database)
    evaluation = InMemoryEvaluation.from_samples_with_truth(
        results, last_train_period
    )

    # Use same metric computation as REST API
    metrics_df = evaluation.compute_metrics([RMSE(), MAE(), CRPS()])

    # Export to CSV
    evaluation.to_flat_forecasts().to_csv("forecasts.csv")
    metrics_df.to_csv("metrics.csv")

    # Optionally persist to database
    if persist:
        backtest_db = evaluation.to_backtest(session, backtest_info)
```

### Example 4: Comparing Evaluations

```python
def compare_evaluations(eval1: EvaluationBase, eval2: EvaluationBase):
    """
    Compare two evaluations regardless of their underlying implementation.
    Works with BacktestEvaluation, InMemoryEvaluation, or any future implementation.
    """
    # Check compatibility
    assert eval1.get_org_units() == eval2.get_org_units()
    assert eval1.get_split_periods() == eval2.get_split_periods()

    # Compute metrics for both
    metrics1 = eval1.compute_metrics([RMSE(), CRPS()])
    metrics2 = eval2.compute_metrics([RMSE(), CRPS()])

    # Compare results
    comparison = pd.merge(
        metrics1, metrics2,
        on=["location", "time_period", "metric_name"],
        suffixes=("_model1", "_model2")
    )
    return comparison

# Usage works with any combination
eval_from_db = BacktestEvaluation.from_backtest(session.get_backtest(1))
eval_from_cli = InMemoryEvaluation.from_samples_with_truth(results, ...)
comparison = compare_evaluations(eval_from_db, eval_from_cli)
```

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      EvaluationBase (ABC)                      │
├────────────────────────────────────────────────────────────────┤
│  + to_flat_forecasts() -> pd.DataFrame                         │
│  + to_flat_observations() -> pd.DataFrame                      │
│  + get_org_units() -> List[str]                                │
│  + get_split_periods() -> List[str]                            │
│  + compute_metrics(metrics) -> pd.DataFrame                    │
└────────────────────────────────────────────────────────────────┘
                               △
                               │ implements
                ┌──────────────┴──────────────┐
                │                             │
┌───────────────┴──────────────┐  ┌──────────┴────────────────────┐
│   BacktestEvaluation         │  │   InMemoryEvaluation          │
├──────────────────────────────┤  ├───────────────────────────────┤
│  - _backtest: BackTest       │  │  - _forecasts: pd.DataFrame   │
│  - _flat_forecasts_cache     │  │  - _observations: pd.DataFrame│
│  - _flat_observations_cache  │  │  - _org_units: List[str]      │
│                              │  │  - _split_periods: List[str]  │
│  + from_backtest()           │  │                               │
│  + to_backtest()             │  │  + from_samples_with_truth()  │
│  + to_flat_forecasts()       │  │  + to_backtest()              │
│  + to_flat_observations()    │  │  + to_flat_forecasts()        │
│  + get_org_units()           │  │  + to_flat_observations()     │
│  + get_split_periods()       │  │  + get_org_units()            │
└──────────────────────────────┘  └───────────────────────────────┘
         │                                   │
         │ wraps                             │ stores
         ▼                                   ▼
┌──────────────────────────────┐  ┌───────────────────────────────┐
│  BackTest (DB Model)         │  │  In-Memory DataFrames         │
│  + forecasts: List[...]      │  │  (FlatForecasts format)       │
│  + dataset: DataSet          │  │                               │
│  + org_units: List[str]      │  │                               │
│  + split_periods: List[str]  │  │                               │
└──────────────────────────────┘  └───────────────────────────────┘
```

## Benefits of This Approach

1. **Code Reuse**: Metric computation, visualization, and analysis code can work with any EvaluationBase implementation

2. **Database Decoupling**: Core evaluation logic no longer depends on database schema

3. **Flexibility**: Easy to add new implementations (e.g., for different storage backends, remote APIs, etc.)

4. **Migration Path**: Can introduce gradually without breaking existing code:
   - Start with BacktestEvaluation wrapping existing BackTest
   - Update visualization/metrics to accept EvaluationBase
   - Later add InMemoryEvaluation for CLI
   - Eventually unify REST API and CLI workflows

5. **Testing**: Easier to test evaluation logic without database setup

6. **Caching**: Implementations can cache expensive conversions (flat representations)

## Future Considerations

### Phase 1: Foundation (No Code Changes)
- Create design document (this document)
- Review and discuss with team
- Get alignment on approach

### Phase 2: Basic Implementation
- Implement EvaluationBase ABC
- Implement BacktestEvaluation wrapper
- Add unit tests

### Phase 3: Gradual Migration
- Update visualization code to accept EvaluationBase
- Update metric computation to accept EvaluationBase
- Ensure backward compatibility with existing code

### Phase 4: CLI Integration
- Implement InMemoryEvaluation
- Refactor CLI evaluate to use InMemoryEvaluation
- Share metric computation code between REST API and CLI

### Phase 5: REST API Refactoring
- Update REST API endpoints to return EvaluationBase
- Simplify worker functions to use abstraction
- Remove direct BackTest manipulation outside database layer

## Open Questions for Discussion

1. **Naming**: Should we use "Evaluation" or keep "Backtest" terminology?
   - Evaluation is more general and not database-specific
   - Backtest is already established in codebase

2. **Aggregate Metrics**: Should EvaluationBase include `get_aggregate_metrics()`?
   - Pro: Matches BackTest schema which has `aggregate_metrics` field
   - Con: Metrics should be computed on-demand, not stored in evaluation

3. **Model Information**: Should Evaluation include model configuration?
   - Currently BackTest has `model_db_id` and `configured_model` relationship
   - For database-agnostic design, should this be optional metadata?

4. **Serialization**: Should we add methods like `to_json()`, `from_json()`?
   - Useful for API responses and caching
   - May be better as separate utility functions

5. **Performance**: Should we optimize for lazy loading or eager loading?
   - BacktestEvaluation caches flat representations
   - InMemoryEvaluation stores them directly
   - Should conversion be done on-demand or upfront?

## Related Files

Key files that would be affected by this refactoring:

- `chap_core/database/tables.py` - BackTest model (unchanged, wrapped by BacktestEvaluation)
- `chap_core/assessment/flat_representations.py` - Conversion functions (reused by implementations)
- `chap_core/assessment/metrics/` - Metric computation (updated to accept EvaluationBase)
- `chap_core/plotting/evaluation_plot.py` - Visualization (updated to accept EvaluationBase)
- `chap_core/rest_api/v1/routers/analytics.py` - REST API endpoints (gradually migrated)
- `chap_core/cli.py` - CLI evaluate command (future integration with InMemoryEvaluation)

## Conclusion

The proposed EvaluationBase abstraction provides a clean separation between evaluation data and storage implementation. By starting with BacktestEvaluation as a wrapper, we can introduce this pattern gradually without breaking changes, then progressively refactor to achieve better code reuse between REST API and CLI workflows.

The key insight is that most evaluation-related code only needs access to flat representations and metadata, not the full database model structure. By defining this interface explicitly, we make dependencies clear and enable more flexible implementations.
