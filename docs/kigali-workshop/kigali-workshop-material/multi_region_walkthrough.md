# Multi-Region Strategies Walkthrough

This walkthrough compares different strategies for handling multiple
geographic regions in a forecasting model. Each strategy represents a
different trade-off between sharing information across regions and
allowing region-specific behavior.

## 1. Setup

```python exec="on" session="regions" source="above"
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.datatypes import Samples
from chap_core.assessment.prediction_evaluator import backtest

dataset = DataSet.from_csv("example_data/laos_subset.csv")
```

We define two estimator classes. `GlobalEstimator` trains a single model
on all regions combined (features can include location information).
`PerRegionEstimator` trains a separate model for each region independently.

```python exec="on" session="regions" source="above"
class GlobalEstimator:
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


class PerRegionEstimator:
    def __init__(self, extract_features):
        self.extract_features = extract_features

    def train(self, data):
        self.models = {}
        for location in data.keys():
            df = data[location].to_pandas().assign(location=location)
            X = self.extract_features(df)
            y = df["disease_cases"].values
            mask = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
            self.models[location] = LinearRegression().fit(X[mask], y[mask])
        return self

    def predict(self, historic_data, future_data):
        results = {}
        for location in future_data.keys():
            hist = historic_data[location].to_pandas().assign(location=location)
            fut = future_data[location].to_pandas().assign(location=location)
            if "disease_cases" not in fut.columns:
                fut["disease_cases"] = np.nan
            combined = pd.concat([hist, fut], ignore_index=True)
            n = len(fut)
            X = self.extract_features(combined).iloc[-n:].fillna(0)
            pred = np.clip(self.models[location].predict(X), 0, None)
            results[location] = Samples(
                future_data[location].time_period, pred.reshape(-1, 1)
            )
        return DataSet(results)
```

```python exec="on" session="regions" source="above"
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

## 2. Strategy: Global Model (Ignoring Regions)

A single model trained on all regions, using only season and climate
as features. The model has no way to distinguish between regions, so
it predicts similar levels for all of them:

```python exec="on" session="regions" source="above" result="text"
def season_climate(df):
    month = pd.get_dummies(df["time_period"].dt.month, prefix="month", dtype=float)
    climate = df[["rainfall", "mean_temperature"]].copy()
    return pd.concat([month, climate], axis=1)


mae = evaluate(GlobalEstimator(season_climate), dataset)
print(f"Global (no location info) MAE: {mae:.1f}")
```

## 3. Strategy: Global Model with Location Offset

Adding location indicator variables gives the model a per-region intercept.
All other effects (season, climate) are still shared across regions:

```python exec="on" session="regions" source="above" result="text"
def location_season_climate(df):
    location = pd.get_dummies(df["location"], dtype=float)
    month = pd.get_dummies(df["time_period"].dt.month, prefix="month", dtype=float)
    climate = df[["rainfall", "mean_temperature"]].copy()
    return pd.concat([location, month, climate], axis=1)


mae = evaluate(GlobalEstimator(location_season_climate), dataset)
print(f"Global + location offset MAE: {mae:.1f}")
```

## 4. Strategy: Separate Model Per Region

Each region gets its own independently fitted model. This allows each
region to have completely different seasonal patterns and climate
responses:

```python exec="on" session="regions" source="above" result="text"
mae = evaluate(PerRegionEstimator(season_climate), dataset)
print(f"Separate per region MAE: {mae:.1f}")
```

## 5. Strategy: Global Model with Location-Specific Seasonality

An intermediate approach: use a single global model, but create interaction
features between location and month. This gives each region its own seasonal
pattern while sharing climate effects:

```python exec="on" session="regions" source="above" result="text"
def location_x_season_climate(df):
    location = pd.get_dummies(df["location"], dtype=float)
    month = pd.get_dummies(df["time_period"].dt.month, prefix="month", dtype=float)
    climate = df[["rainfall", "mean_temperature"]].copy()
    interactions = pd.DataFrame(index=df.index)
    for loc_col in location.columns:
        for month_col in month.columns:
            interactions[f"{loc_col}_{month_col}"] = location[loc_col] * month[month_col]
    return pd.concat([location, climate, interactions], axis=1)


mae = evaluate(GlobalEstimator(location_x_season_climate), dataset)
print(f"Location x season MAE: {mae:.1f}")
```

## 6. Discussion

With only 3 regions and 36 months of data, separate per-region models
perform well -- each region has enough data to fit the simple model
reliably. In datasets with many regions and less data per region, the
balance shifts: separate models overfit, and shared approaches that
"borrow strength" across regions become important.

Hierarchical (partial pooling) models offer a principled middle ground.
Instead of fully sharing or fully separating parameters, they allow each
region's parameters to deviate from a shared mean, with the amount of
deviation learned from data. This is the primary use case for hierarchical
Bayesian models, which CHAP supports through frameworks like PyMC.
