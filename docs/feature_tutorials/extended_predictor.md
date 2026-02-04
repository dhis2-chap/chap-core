# Extended Predictor

## Overview

The `ExtendedPredictor` class enables models to predict beyond their maximum prediction length by using an iterative prediction strategy. This is useful when you need forecasts for a longer time horizon than what a model natively supports.

## How It Works

The `ExtendedPredictor` wraps a `ConfiguredModel` and extends its prediction capability through the following approach:

1. **Iterative Prediction**: When the desired prediction scope exceeds the model's `max_prediction_length`, the predictor makes multiple prediction calls in sequence.

2. **Rolling History Update**: After each prediction step, the predicted values are appended to the historic data. Sample columns are averaged to produce a single `disease_cases` value for use in subsequent predictions.

3. **Overlap Handling**: When predictions overlap (which happens in later iterations), duplicates are removed by keeping the most recent prediction for each time period and location.

The core logic is in the `predict` method, which loops until the full desired scope is covered:

```

while remaining_time_periods > 0:
    steps_to_predict = min(max_pred_length, remaining_time_periods)
    # ... make prediction for steps_to_predict periods
    # ... update historic data with predictions
    remaining_time_periods -= newly_predicted
    
```

## Testing the Functionality

To verify that `ExtendedPredictor` works correctly, run an evaluation with a prediction length that exceeds the model's native maximum. The `eval` command automatically wraps models with `ExtendedPredictor` when needed:

```bash
chap eval --model-name external_models/naive_python_model_uv \
    --dataset-csv example_data/laos_subset.csv \
    --output-file ./extended_predictor_test.nc \
    --backtest-params.n-periods 6 \
    --backtest-params.n-splits 2
```

```bash
rm -f ./extended_predictor_test.nc
```

When the requested `n-periods` exceeds the model's `max_prediction_length`, CHAP automatically uses `ExtendedPredictor` to make iterative predictions.

> **Note:** The legacy `chap evaluate` command is deprecated and will be removed in v2.0. Use `chap eval` instead.
