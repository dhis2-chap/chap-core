# Progressive Effects Walkthrough

This walkthrough shows how to progressively add modeling effects to a simple
linear regression. Each step adds a new type of feature and we measure
improvement via backtesting.

By the end, you will have built a model with location-specific offsets,
seasonal patterns, climate covariates, and lagged disease cases.

## 1. Loading the Data

```python exec="on" session="effects" source="above"
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

dataset = DataSet.from_csv("example_data/laos_subset.csv")
```

```python exec="on" session="effects" source="above" result="text"
print("Locations:", list(dataset.keys()))
print("Period range:", dataset.period_range)
print("Number of periods:", len(dataset.period_range))
```

## 2. A Basic Estimator

We define a `BasicEstimator` that takes a feature extraction function.
Different feature functions produce different models, while the estimator
handles the training and prediction boilerplate.

```python exec="on" session="effects" source="above"
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from chap_core.datatypes import Samples


class BasicEstimator:
    def __init__(self, extract_features):
        self.extract_features = extract_features

    def train(self, data):
        df = data.to_pandas()
        X = self.extract_features(df)
        y = df["disease_cases"].values
        mask = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
        self.model = LinearRegression().fit(X[mask], y[mask])
        return self

    def predict(self, historic_data, future_data):
        parts, future_mask = [], []
        for location in future_data.keys():
            hist = historic_data[location].to_pandas().assign(location=location)
            fut = future_data[location].to_pandas().assign(location=location)
            if "disease_cases" not in fut.columns:
                fut["disease_cases"] = np.nan
            parts.append(pd.concat([hist, fut], ignore_index=True))
            future_mask += [False] * len(hist) + [True] * len(fut)
        combined = pd.concat(parts, ignore_index=True)
        X = self.extract_features(combined).fillna(0)
        pred = np.clip(self.model.predict(X[future_mask]), 0, None)
        results, i = {}, 0
        for location in future_data.keys():
            n = len(future_data[location])
            results[location] = Samples(
                future_data[location].time_period, pred[i : i + n].reshape(-1, 1)
            )
            i += n
        return DataSet(results)
```

The `predict` method combines all locations' historic and future data into a
single DataFrame before extracting features. This ensures feature columns
(like location dummies) stay consistent between training and prediction, and
allows lag-based features to look back into the historic window.

## 3. Evaluation Helper

We use `backtest` to run expanding-window cross-validation and compute
mean absolute error (MAE) for each model variant:

```python exec="on" session="effects" source="above"
from chap_core.assessment.prediction_evaluator import backtest


def evaluate(estimator, dataset, prediction_length=3, n_test_sets=4):
    results = list(backtest(
        estimator, dataset,
        prediction_length=prediction_length, n_test_sets=n_test_sets,
    ))
    errors = []
    for result in results:
        for location in result.keys():
            truth = result[location].disease_cases
            predicted = result[location].samples[:, 0]
            errors.extend(np.abs(truth - predicted))
    return np.mean(errors)
```

## 4. Location-Specific Offset

The simplest region-aware feature: one indicator variable per location.
This lets the model learn a different baseline for each region.

```python exec="on" session="effects" source="above" result="text"
def location_offset(df):
    return pd.get_dummies(df["location"], dtype=float)


mae = evaluate(BasicEstimator(location_offset), dataset)
print(f"Location offset MAE: {mae:.1f}")
```

## 5. Seasonal Effect

Disease incidence often follows seasonal patterns. Adding month-of-year
indicators captures periodic variation:

```python exec="on" session="effects" source="above" result="text"
def location_and_season(df):
    location = pd.get_dummies(df["location"], dtype=float)
    month = pd.get_dummies(df["time_period"].dt.month, prefix="month", dtype=float)
    return pd.concat([location, month], axis=1)


mae = evaluate(BasicEstimator(location_and_season), dataset)
print(f"Location + season MAE: {mae:.1f}")
```

## 6. Climate Covariates

CHAP provides future climate data (rainfall, temperature) at prediction time,
so we can use these as features directly. This captures the relationship
between climate conditions and disease incidence:

```python exec="on" session="effects" source="above" result="text"
def location_season_climate(df):
    location = pd.get_dummies(df["location"], dtype=float)
    month = pd.get_dummies(df["time_period"].dt.month, prefix="month", dtype=float)
    climate = df[["rainfall", "mean_temperature"]].copy()
    return pd.concat([location, month, climate], axis=1)


mae = evaluate(BasicEstimator(location_season_climate), dataset)
print(f"Location + season + climate MAE: {mae:.1f}")
```

In practice, climate effects on disease are often delayed (e.g. rainfall
affects mosquito breeding over weeks). You can also add lagged climate
features using `df.groupby("location")["rainfall"].shift(lag)`, but with
limited data, adding many lag features risks overfitting.

## 7. Lagged Target (Disease Cases)

Past disease cases are typically the strongest predictor of future cases.
However, lagged target introduces a technical difficulty: at prediction
time, future disease cases are unknown.

The simplest solution is to only use lags at least as long as the
forecast horizon. Since we predict 3 months ahead, lag 3 is the shortest
usable lag -- its value is always known at prediction time.

```python exec="on" session="effects" source="above" result="text"
def all_features(df):
    location = pd.get_dummies(df["location"], dtype=float)
    month = pd.get_dummies(df["time_period"].dt.month, prefix="month", dtype=float)
    climate = df[["rainfall", "mean_temperature"]].copy()
    lags = pd.DataFrame(index=df.index)
    lags["cases_lag3"] = df.groupby("location")["disease_cases"].shift(3)
    return pd.concat([location, month, climate, lags], axis=1)


mae = evaluate(BasicEstimator(all_features), dataset)
print(f"All features MAE: {mae:.1f}")
```

Using shorter lags (e.g. lag 1 or 2) would require recursive forecasting:
predicting one step ahead, feeding that prediction back as input, then
predicting the next step. This is more complex to implement and can
accumulate errors across steps.
