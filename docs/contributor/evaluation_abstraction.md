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

### Core Concept: FlatEvaluationData

First, we define a simple dataclass that combines forecasts and observations together:

```python
from dataclasses import dataclass
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved

@dataclass
class FlatEvaluationData:
    """
    Container for flat representations of evaluation data.

    Combines forecasts and observations which are always used together
    for metric computation and visualization.
    """
    forecasts: FlatForecasts
    observations: FlatObserved
```

### Core Concept: EvaluationBase ABC

Create an abstract base class that defines the interface for all evaluation representations, decoupled from database implementation:

```python
from abc import ABC, abstractmethod
from typing import List

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
    def to_flat(self) -> FlatEvaluationData:
        """
        Export evaluation data as flat representations.

        Returns:
            FlatEvaluationData containing FlatForecasts and FlatObserved objects
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

    @classmethod
    @abstractmethod
    def from_backtest(cls, backtest: "BackTest") -> "EvaluationBase":
        """
        Create Evaluation from database BackTest object.

        All implementations must support loading from database.

        Args:
            backtest: Database BackTest object (with relationships loaded)

        Returns:
            Evaluation instance
        """
        pass
```

### Concrete Implementation: Evaluation

Wraps existing BackTest database model to implement the abstract interface:

```python
class Evaluation(EvaluationBase):
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
        self._flat_data_cache = None

    @classmethod
    def from_backtest(cls, backtest: BackTest) -> "Evaluation":
        """
        Create Evaluation from database BackTest object.

        Args:
            backtest: Database BackTest object (with relationships loaded)

        Returns:
            Evaluation instance
        """
        return cls(backtest)

    def to_backtest(self) -> BackTest:
        """
        Get underlying database BackTest object.

        Returns:
            BackTest database model
        """
        return self._backtest

    def to_flat(self) -> FlatEvaluationData:
        """Export evaluation data using existing conversion functions."""
        if self._flat_data_cache is None:
            from chap_core.assessment.flat_representations import (
                FlatForecasts,
                FlatObserved,
                convert_backtest_to_flat_forecasts,
                convert_backtest_observations_to_flat_observations,
            )

            forecasts_df = convert_backtest_to_flat_forecasts(
                self._backtest.forecasts
            )
            observations_df = convert_backtest_observations_to_flat_observations(
                self._backtest.dataset.observations
            )

            self._flat_data_cache = FlatEvaluationData(
                forecasts=FlatForecasts(forecasts_df),
                observations=FlatObserved(observations_df),
            )
        return self._flat_data_cache

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
        flat_data: FlatEvaluationData,
        org_units: List[str],
        split_periods: List[str],
    ):
        """
        Args:
            flat_data: FlatEvaluationData containing forecasts and observations
            org_units: List of location identifiers
            split_periods: List of split period identifiers
        """
        self._flat_data = flat_data
        self._org_units = org_units
        self._split_periods = split_periods

    @classmethod
    def from_backtest(cls, backtest: BackTest) -> "InMemoryEvaluation":
        """
        Create InMemoryEvaluation from database BackTest object.

        Converts database representation to in-memory format.

        Args:
            backtest: Database BackTest object (with relationships loaded)

        Returns:
            InMemoryEvaluation instance
        """
        from chap_core.assessment.flat_representations import (
            FlatForecasts,
            FlatObserved,
            convert_backtest_to_flat_forecasts,
            convert_backtest_observations_to_flat_observations,
        )

        forecasts_df = convert_backtest_to_flat_forecasts(backtest.forecasts)
        observations_df = convert_backtest_observations_to_flat_observations(
            backtest.dataset.observations
        )

        flat_data = FlatEvaluationData(
            forecasts=FlatForecasts(forecasts_df),
            observations=FlatObserved(observations_df),
        )

        return cls(
            flat_data=flat_data,
            org_units=backtest.org_units,
            split_periods=backtest.split_periods,
        )

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

    def to_flat(self) -> FlatEvaluationData:
        """Return flat data directly."""
        return self._flat_data

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
# Compute metrics manually
for metric in [RMSE(), MAE(), CRPS()]:
    metric_df = metric.get_metric(flat_observations, flat_forecasts)

# Proposed approach (using abstraction)
backtest_db = session.get_backtest(backtest_id)
evaluation = Evaluation.from_backtest(backtest_db)

# Get flat data for metric computation
flat_data = evaluation.to_flat()
for metric in [RMSE(), MAE(), CRPS()]:
    metric_df = metric.get_metric(flat_data.observations, flat_data.forecasts)

# Access metadata
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
    flat_data = evaluation.to_flat()
    metric_data = metric.compute(flat_data.observations, flat_data.forecasts)
    return plot(metric_data)

# Usage
evaluation = Evaluation.from_backtest(backtest_db)
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

    # Get flat data
    flat_data = evaluation.to_flat()

    # Use same metric computation as REST API
    for metric in [RMSE(), MAE(), CRPS()]:
        metric_df = metric.get_metric(flat_data.observations, flat_data.forecasts)
        # Process metric_df...

    # Export to CSV (accessing underlying DataFrames)
    flat_data.forecasts.to_csv("forecasts.csv")
    flat_data.observations.to_csv("observations.csv")

    # Optionally persist to database
    if persist:
        backtest_db = evaluation.to_backtest(session, backtest_info)
```

### Example 4: Comparing Evaluations

```python
def compare_evaluations(eval1: EvaluationBase, eval2: EvaluationBase):
    """
    Compare two evaluations regardless of their underlying implementation.
    Works with Evaluation, InMemoryEvaluation, or any future implementation.
    """
    # Check compatibility
    assert eval1.get_org_units() == eval2.get_org_units()
    assert eval1.get_split_periods() == eval2.get_split_periods()

    # Get flat data for both
    flat_data1 = eval1.to_flat()
    flat_data2 = eval2.to_flat()

    # Compute same metrics for both
    results1 = {}
    results2 = {}
    for metric in [RMSE(), CRPS()]:
        results1[metric.spec.metric_id] = metric.get_metric(
            flat_data1.observations, flat_data1.forecasts
        )
        results2[metric.spec.metric_id] = metric.get_metric(
            flat_data2.observations, flat_data2.forecasts
        )

    # Compare results (implementation varies by metric output format)
    return results1, results2

# Usage works with any combination
# Both implementations support from_backtest()
eval1 = Evaluation.from_backtest(session.get_backtest(1))
eval2 = InMemoryEvaluation.from_backtest(session.get_backtest(2))
results1, results2 = compare_evaluations(eval1, eval2)

# Or use from_samples_with_truth() for CLI results
eval_from_cli = InMemoryEvaluation.from_samples_with_truth(results, ...)
results_cli, results_db = compare_evaluations(eval_from_cli, eval1)
```

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                  FlatEvaluationData (dataclass)                │
├────────────────────────────────────────────────────────────────┤
│  + forecasts: FlatForecasts                                    │
│  + observations: FlatObserved                                  │
└────────────────────────────────────────────────────────────────┘
                               △
                               │ returns
                               │
┌────────────────────────────────────────────────────────────────┐
│                      EvaluationBase (ABC)                      │
├────────────────────────────────────────────────────────────────┤
│  + to_flat() -> FlatEvaluationData                             │
│  + get_org_units() -> List[str]                                │
│  + get_split_periods() -> List[str]                            │
│  + from_backtest(backtest) -> EvaluationBase [classmethod]     │
└────────────────────────────────────────────────────────────────┘
                               △
                               │ implements
                ┌──────────────┴──────────────┐
                │                             │
┌───────────────┴──────────────┐  ┌──────────┴────────────────────┐
│   Evaluation         │  │   InMemoryEvaluation          │
├──────────────────────────────┤  ├───────────────────────────────┤
│  - _backtest: BackTest       │  │  - _flat_data:                │
│  - _flat_data_cache          │  │      FlatEvaluationData       │
│                              │  │  - _org_units: List[str]      │
│                              │  │  - _split_periods: List[str]  │
│  + from_backtest()           │  │                               │
│  + to_backtest()             │  │  + from_samples_with_truth()  │
│  + to_flat()                 │  │  + to_backtest()              │
│  + get_org_units()           │  │  + to_flat()                  │
│  + get_split_periods()       │  │  + get_org_units()            │
│                              │  │  + get_split_periods()        │
└──────────────────────────────┘  └───────────────────────────────┘
         │                                   │
         │ wraps                             │ stores
         ▼                                   ▼
┌──────────────────────────────┐  ┌───────────────────────────────┐
│  BackTest (DB Model)         │  │  FlatEvaluationData           │
│  + forecasts: List[...]      │  │  (in-memory)                  │
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
   - Start with Evaluation wrapping existing BackTest
   - Update visualization/metrics to accept EvaluationBase
   - Later add InMemoryEvaluation for CLI
   - Eventually unify REST API and CLI workflows

5. **Testing**: Easier to test evaluation logic without database setup

6. **Caching**: Implementations can cache expensive conversions (flat representations)

## Implementation Strategy

The implementation will be done in phases to minimize risk and allow for incremental progress. **Phase 1 is the immediate next step** - implementing the Evaluation classes without changing any existing code.

---

## Implementation Phases

### Phase 0: Design (Current Phase)
**Goal**: Document the design and get team alignment

**Tasks**:
- Create design document (this document)
- Review and discuss with team
- Get alignment on approach
- Refine design based on feedback

**Deliverable**: Approved design document

---

### Phase 1: Core Implementation (First Step - Keep It Simple)
**Goal**: Implement the Evaluation abstraction without changing any existing code

**Scope**: Create new classes only - no refactoring of existing code

**New File**: `chap_core/assessment/evaluation.py`

**Classes to Implement**:

1. **`FlatEvaluationData`** (dataclass):
   ```python
   @dataclass
   class FlatEvaluationData:
       forecasts: FlatForecasts
       observations: FlatObserved
   ```

2. **`EvaluationBase`** (ABC):
   - Abstract method: `to_flat() -> FlatEvaluationData`
   - Abstract method: `get_org_units() -> List[str]`
   - Abstract method: `get_split_periods() -> List[str]`
   - Abstract classmethod: `from_backtest(backtest) -> EvaluationBase`

3. **`Evaluation`** (concrete implementation):
   - Constructor: `__init__(self, backtest: BackTest)`
   - Classmethod: `from_backtest(backtest) -> Evaluation`
   - Method: `to_backtest() -> BackTest` (return wrapped object)
   - Method: `to_flat() -> FlatEvaluationData` (with caching)
   - Method: `get_org_units() -> List[str]`
   - Method: `get_split_periods() -> List[str]`

**Testing**:
- Create `tests/test_evaluation.py`
- Test `Evaluation.from_backtest()` with mock BackTest
- Test `to_flat()` returns correct types and data
- Test metadata accessors work correctly
- Verify conversion matches existing `convert_backtest_to_flat_*()` functions

**What we do NOT do in Phase 1**:
- ❌ Change any existing REST API code
- ❌ Change any existing CLI code
- ❌ Change any visualization or metric computation code
- ❌ Change the database schema
- ❌ Implement InMemoryEvaluation (that's Phase 3)

**Success Criteria**:
- All tests pass
- Can create `Evaluation` from database `BackTest` object
- Can convert to flat representations correctly
- Code is documented with docstrings
- No existing code is modified

**Deliverable**: New `evaluation.py` module with working, tested classes that can load from database but aren't yet used anywhere

---

### Phase 2: REST API Integration
**Goal**: Refactor REST API to use the Evaluation abstraction

**Tasks**:
1. Update analytics router endpoints to work with `EvaluationBase`
2. Update worker functions to optionally return `Evaluation`
3. Update metric computation functions to accept `EvaluationBase`
4. Update visualization functions to accept `EvaluationBase`
5. Ensure backward compatibility throughout
6. Add integration tests

**Files to modify**:
- `chap_core/rest_api/v1/routers/analytics.py`
- `chap_core/rest_api/db_worker_functions.py`
- `chap_core/assessment/metrics/__init__.py`
- `chap_core/plotting/evaluation_plot.py`

**Deliverable**: REST API using Evaluation abstraction while maintaining all existing functionality

---

### Phase 3: CLI Integration
**Goal**: Implement InMemoryEvaluation and refactor CLI to use it

**Tasks**:
1. Implement `InMemoryEvaluation` class:
   - Implements `from_backtest()` for loading from DB
   - Implements `from_samples_with_truth()` for CLI workflow
   - Implements `to_backtest()` for optional persistence
2. Refactor `cli.py` evaluate command to use `InMemoryEvaluation`
3. Share metric computation code between REST API and CLI
4. Add CLI-specific tests

**Deliverable**: CLI and REST API using same evaluation abstraction and metric computation

---

### Phase 4: Code Consolidation
**Goal**: Remove duplication and clean up deprecated code

**Tasks**:
1. Identify and remove duplicated evaluation logic
2. Consolidate metric computation into shared utilities
3. Update documentation and examples
4. Remove deprecated functions if any
5. Performance optimization if needed

**Deliverable**: Cleaner codebase with less duplication and better maintainability

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
   - Evaluation caches flat representations
   - InMemoryEvaluation stores them directly
   - Should conversion be done on-demand or upfront?

## Related Files

Key files that would be affected by this refactoring:

- `chap_core/database/tables.py` - BackTest model (unchanged, wrapped by Evaluation)
- `chap_core/assessment/flat_representations.py` - Conversion functions (reused by implementations)
- `chap_core/assessment/metrics/` - Metric computation (updated to accept EvaluationBase)
- `chap_core/plotting/evaluation_plot.py` - Visualization (updated to accept EvaluationBase)
- `chap_core/rest_api/v1/routers/analytics.py` - REST API endpoints (gradually migrated)
- `chap_core/cli.py` - CLI evaluate command (future integration with InMemoryEvaluation)

## Conclusion

The proposed EvaluationBase abstraction provides a clean separation between evaluation data and storage implementation. By starting with Evaluation as a wrapper, we can introduce this pattern gradually without breaking changes, then progressively refactor to achieve better code reuse between REST API and CLI workflows.

The key insight is that most evaluation-related code only needs access to flat representations and metadata, not the full database model structure. By defining this interface explicitly, we make dependencies clear and enable more flexible implementations.
