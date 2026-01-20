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

To verify that `ExtendedPredictor` works correctly, run the following evaluation command:

```bash
chap evaluate --model-name https://github.com/chap-models/Xiang_SVM --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil
```

This command evaluates the Xiang SVM model on the ISIMIP dengue dataset for Brazil, which triggers the extended prediction logic when the evaluation requires predictions beyond the model's native maximum prediction length.
