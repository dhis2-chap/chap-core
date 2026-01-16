# Chap's broad range of supporting functionality for modelling

Chap already contains rich functionality for data handling, data input from DHIS2 and rigorous model evaluation.
This allows modellers to focus only on training and predicting based on a single provided dataset,
while relying on the Chap framework to collect data from various sources, parse data of different formats, and perform multiple train-and-predict iterations as part of a rigorous time-series cross-validation.

## Extended prediction horizons

If you specify `max_prediction_length` in your model configuration, CHAP can automatically extend your model's prediction horizon using ExtendedPredictor. This wrapper makes predictions beyond your model's limit through iterative prediction: it predicts up to the maximum length, adds those predictions to the historic data, then predicts again. This repeats until the desired forecast length is reached.

There are ongoing developments for a range of further supporting features, allowing modellers to rely on the Chap framework for model tuning (autoML), ensemble model learning, model explainability and more (please see overview of planned features [here](https://github.com/dhis2-chap/chap-core/wiki/CHAP-Roadmap))   