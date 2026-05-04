# Native SHAP Values

If your model can compute SHAP values directly — for example because it is tree-based or linear — you can output them alongside predictions. CHAP will store these native SHAP values and surface them in the Explainability Widget as global feature importance, beeswarm plots, waterfall charts, and horizon summaries, without fitting a surrogate model.

This is purely opt-in. Models that do not provide SHAP values can use surrogate-based explanations instead.

## Step 1: Declare native SHAP support in MLproject

Add `provides_native_shap: true` to your `MLproject` file:

```yaml
name: my_model

python_env: python_env.yaml

provides_native_shap: true

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python model.py train {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python model.py predict {model} {historic_data} {future_data} {out_file}"
```

Without this flag CHAP ignores any `shap_values.csv` file your model writes, so the flag is required.

## Step 2: Write `shap_values.csv` during predict

Your predict entry point must write a file named `shap_values.csv` to the current working directory alongside `predictions.csv`.

Preferred format (supports engineered features with explicit values):

```
location,time_period,expected_value,shap__rainfall,shap__mean_temperature,value__rainfall,value__mean_temperature
district_A,2024-01,105.2,0.52,-0.31,87.4,28.3
district_A,2024-02,110.8,0.61,-0.25,91.2,29.1
district_B,2024-01,88.3,-0.15,0.42,54.0,31.7
```

Column rules:

- `location` and `time_period` must match the corresponding rows in `predictions.csv`.
- `expected_value` is the SHAP baseline value `E[f(x)]` for that row (the model's average prediction, or the intercept for additive models). Each row can have its own value, or you can repeat a constant baseline.
- SHAP columns must be prefixed as `shap__<feature_name>`.
- Feature-value columns are required: include `value__<feature_name>` for every SHAP column. This allows UI popups to show the true feature values, especially for engineered features not present in CHAP's covariate dataset.
- Row order does not need to match `predictions.csv`. CHAP joins on `(location, time_period)`.
- If `shap_values.csv` is missing or cannot be parsed, CHAP logs a warning and native SHAP data is not stored for this prediction. Surrogate-based methods (e.g. `shap_auto`) remain available but must be requested explicitly.

## Python example

This example uses a scikit-learn `GradientBoostingRegressor` with `shap`:

```console
import pandas as pd
import shap

# Assume `model` is a fitted GradientBoostingRegressor,
# `X_future` is the feature DataFrame for forecast rows,
# and `future_df` already has `location` and `time_period` columns.

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_future)   # shape (n_rows, n_features)
expected_value = explainer.expected_value        # scalar baseline

shap_df = pd.DataFrame(shap_values, columns=[f"shap__{c}" for c in X_future.columns])
for c in X_future.columns:
    shap_df[f"value__{c}"] = X_future[c].values
shap_df.insert(0, "time_period", future_df["time_period"].values)
shap_df.insert(0, "location", future_df["location"].values)
shap_df.insert(2, "expected_value", expected_value)

shap_df.to_csv("shap_values.csv", index=False)
```

For a linear model the pattern is the same but use `shap.LinearExplainer`:

```console
import shap

explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_future)
expected_value = explainer.expected_value
```

## R example

This example uses the `treeshap` package with a `ranger` random forest:

```r
library(ranger)
library(treeshap)

# Assume `model` is a fitted ranger object,
# `future_df` has location, time_period, and feature columns.

feature_cols <- setdiff(colnames(future_df), c("location", "time_period", "disease_cases"))
X_future <- future_df[, feature_cols]

unified  <- ranger.unify(model, X_future)
shap_out <- treeshap(unified, X_future)

shap_df <- as.data.frame(shap_out$shaps)
colnames(shap_df) <- paste0("shap__", feature_cols)
for (c in feature_cols) {
  shap_df[[paste0("value__", c)]] <- X_future[[c]]
}

shap_df$location       <- future_df$location
shap_df$time_period    <- future_df$time_period
shap_df$expected_value <- mean(model$predictions)   # training-set mean as baseline

# reorder columns
shap_cols <- paste0("shap__", feature_cols)
value_cols <- paste0("value__", feature_cols)
shap_df <- shap_df[, c("location", "time_period", "expected_value", shap_cols, value_cols)]

write.csv(shap_df, "shap_values.csv", row.names = FALSE)
```

## What the user sees

When CHAP detects native SHAP values for a prediction it:

- Exposes the `native_shap` method via the XAI API (visible in `/xai/methods` and the Explainability Widget method selector).
- Replaces the surrogate quality panel (R-squared, MAE) with a **"Direct from model"** indicator, because no surrogate approximation is involved.
- Renders the same visualisations as surrogate SHAP: global feature importance, beeswarm plot, waterfall chart (local), and horizon summary.

Global and local explanations for `native_shap` are derived directly from the stored `shap_values.csv` data; no surrogate is trained. The actual prediction for a row is computed as `expected_value + sum(shap_values)` for that row.

Users can still call surrogate-based methods (for example `shap_auto`) alongside native SHAP to compare results.

## Checklist

- `provides_native_shap: true` in `MLproject`
- `shap_values.csv` written to the working directory by the predict entry point
- Columns: `location`, `time_period`, `expected_value`, `shap__<feature>` columns, and `value__<feature>` columns (required)
- `location` and `time_period` values match those in `predictions.csv`
- SHAP values are per-row, not aggregated (CHAP derives global summaries from them)

