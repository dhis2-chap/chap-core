# Evaluation Walkthrough

This walkthrough is for educational purposes. It breaks the evaluation pipeline
into individual steps so you can see what happens at each stage. In practice,
use the higher-level `Evaluation.create` (section 7) or the CLI `chap eval`
command rather than calling the lower-level splitting and prediction functions
directly.

For the conceptual overview and architecture diagrams, see
[Evaluation Pipeline](evaluation_pipeline.md).

## 1. Loading a Dataset

A `DataSet` is the central data structure in Chap. It maps location names to
typed time-series arrays. Load one from CSV:

```python
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

dataset = DataSet.from_csv("example_data/laos_subset.csv")
```

Inspect locations, time range, and available fields:

```python
import dataclasses

print(list(dataset.keys()))
print(dataset.period_range)
print(len(dataset.period_range))

location = list(dataset.keys())[0]
field_names = [f.name for f in dataclasses.fields(dataset[location])]
print(field_names)
```

```text
['Bokeo', 'Savannakhet', 'Vientiane[prefecture]']
PeriodRange(Month(2010-1)..Month(2012-12))
36
['time_period', 'rainfall', 'mean_temperature', 'disease_cases', 'population']
```

Each location holds arrays for `time_period`, `rainfall`, `mean_temperature`,
`disease_cases`, and `population`.

## 2. Splitting the Data

The `train_test_generator` function implements expanding-window cross-validation.
It returns a training set and an iterator of `(historic, masked_future, future_truth)`
tuples.

```python
from chap_core.assessment.dataset_splitting import train_test_generator

train_set, splits = train_test_generator(
    dataset, prediction_length=3, n_test_sets=4, stride=1
)
splits = list(splits)
```

The training set covers the earliest portion of the data:

```python
print(train_set.period_range)
print(len(train_set.period_range))
```

```text
PeriodRange(Month(2010-1)..Month(2012-6))
30
```

Each split provides three datasets per location:

- **historic_data** -- all data up to the split point (grows each split)
- **masked_future_data** -- future covariates *without* `disease_cases`
- **future_data** -- full future data including `disease_cases` (ground truth)

```python
for i, (historic, masked_future, future_truth) in enumerate(splits):
    print(
        f"Split {i}: historic periods={len(historic.period_range)}, "
        f"future range={future_truth.period_range}"
    )
```

```text
Split 0: historic periods=30, future range=PeriodRange(Month(2012-7)..Month(2012-9))
Split 1: historic periods=31, future range=PeriodRange(Month(2012-8)..Month(2012-10))
Split 2: historic periods=32, future range=PeriodRange(Month(2012-9)..Month(2012-11))
Split 3: historic periods=33, future range=PeriodRange(Month(2012-10)..Month(2012-12))
```

## 3. How Test Instances Differ

The historic window expands by `stride` periods with each successive split, while
the future window slides forward:

```python
for i, (historic, masked_future, future_truth) in enumerate(splits):
    print(
        f"Split {i}: historic={len(historic.period_range)} periods, "
        f"future starts at {future_truth.period_range[0]}"
    )
```

```text
Split 0: historic=30 periods, future starts at Month(2012-7)
Split 1: historic=31 periods, future starts at Month(2012-8)
Split 2: historic=32 periods, future starts at Month(2012-9)
Split 3: historic=33 periods, future starts at Month(2012-10)
```

The masked future data has climate features but no `disease_cases`, which is
exactly what a model receives at prediction time:

```python
location = list(splits[0][1].keys())[0]
masked_fields = [f.name for f in dataclasses.fields(splits[0][1][location])]
print(masked_fields)
```

```text
['time_period', 'rainfall', 'mean_temperature', 'population']
```

## 4. Running a Prediction on a Test Instance

Train the `NaiveEstimator` (which predicts Poisson samples around each location's
historical mean) and predict on one split:

```python
from chap_core.predictor.naive_estimator import NaiveEstimator

estimator = NaiveEstimator()
predictor = estimator.train(train_set)

historic, masked_future, future_truth = splits[0]
predictions = predictor.predict(historic, masked_future)
```

The result is a `DataSet[Samples]` -- each location holds a 2D array of shape
`(n_periods, n_samples)`:

```python
location = list(predictions.keys())[0]
print(predictions[location].samples.shape)
```

```text
(3, 100)
```

## 5. Comparing Predictions to Truth

Merge predictions with ground truth using `DataSet.merge`:

```python
from chap_core.datatypes import SamplesWithTruth
import numpy as np

merged = future_truth.merge(predictions, result_dataclass=SamplesWithTruth)

location = list(merged.keys())[0]
print("Observed:", merged[location].disease_cases)
print("Predicted median:", np.median(merged[location].samples, axis=1))
```

```text
Observed: [17. 17. 26.]
Predicted median: [5. 6. 5.]
```

Each `SamplesWithTruth` entry pairs the observed `disease_cases` with the
predicted `samples` array, enabling metric computation.

## 6. Running a Full Backtest

The `backtest` function ties sections 2-5 together: it splits the data, trains
the model once, predicts for each split, and merges with ground truth.

```python
from chap_core.assessment.prediction_evaluator import backtest

results = list(backtest(estimator, dataset, prediction_length=3, n_test_sets=4, stride=1))
print(f"{len(results)} splits")

for i, result in enumerate(results):
    print(f"Split {i}: periods={result.period_range}")
```

```text
4 splits
Split 0: periods=PeriodRange(Month(2012-7)..Month(2012-9))
Split 1: periods=PeriodRange(Month(2012-8)..Month(2012-10))
Split 2: periods=PeriodRange(Month(2012-9)..Month(2012-11))
Split 3: periods=PeriodRange(Month(2012-10)..Month(2012-12))
```

Each result is a `DataSet[SamplesWithTruth]` covering all locations for one
test window.

## 7. Creating an Evaluation Object

`Evaluation.create` wraps the full backtest workflow and produces an object that
supports export to flat DataFrames and NetCDF files.

The `NaiveEstimator` provides `model_template_db` and `configured_model_db` class
attributes with the model metadata needed by the evaluation:

Run the evaluation:

```python
from chap_core.api_types import BacktestParams
from chap_core.assessment.evaluation import Evaluation

backtest_params = BacktestParams(n_periods=3, n_splits=4, stride=1)
evaluation = Evaluation.create(estimator.configured_model_db, estimator, dataset, backtest_params)
```

Export to flat DataFrames for inspection:

```python
import pandas as pd

flat = evaluation.to_flat()

forecasts_df = pd.DataFrame(flat.forecasts)
observations_df = pd.DataFrame(flat.observations)

print(forecasts_df.head().to_markdown())
```

|    | location   |   time_period |   horizon_distance |   sample |   forecast |
|---:|:-----------|--------------:|-------------------:|---------:|-----------:|
|  0 | Bokeo      |        201207 |                  0 |        0 |          3 |
|  1 | Bokeo      |        201207 |                  0 |        1 |          8 |
|  2 | Bokeo      |        201207 |                  0 |        2 |          7 |
|  3 | Bokeo      |        201207 |                  0 |        3 |          5 |
|  4 | Bokeo      |        201207 |                  0 |        4 |          5 |

Export to a NetCDF file for sharing or later analysis:

```python
import tempfile

with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
    evaluation.to_file(f.name)
    print(f"Saved to {f.name}")
```

```text
Saved to /tmp/tmpXXXXXXXX.nc
```
