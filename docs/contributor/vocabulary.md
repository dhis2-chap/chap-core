# Vocabulary (domain specific terms used in the code)


### Model template

A model template is a flexible "model" which can be configured. A model template typically presents various options (hyperparameters, possible covariates, etc) which are open for configuration. 


### Configured model

A configured model can be made from a model template by applying chocies to the options presented by the model template. Only a configured model can actually be trained on a given dataset (as opposed to a model template, since a model template does not necessarily have enough information about how to train or predict).

### ExternalModel

ExternalModel is a wrapper around an external model (that can e.g. be an R model) to make it compatible with the interface of ConfiguredModel. This means that ExternalModel has train/predict similarily to ConfiguredModel, but these methods are wrappers that runs the train/predict of external models.

## Runner

A runner is something that can run commands. ExternalModels (not ConfiguredModels) have a Runner object attached to them. When train/predict is called, the runner is handling how to j6o05..,m.


## Evaluation and backtesting

Evaluation is the broad term and a backtest is one kind of evaluation. The code and the REST API do not follow this yet: they use "backtest" and "evaluation" for the same thing (for example the `Backtest` table and the `Evaluation` class, which wraps a `Backtest`).

### Evaluation

Finding out how well one or more models work. It is the broad term, used for any assessment of model performance, also when several models are compared side by side (for example baselines and the models of interest). Use "evaluation" rather than "benchmark" or "benchmarking" in new text. In Chap, evaluation is usually done by backtesting and read through metrics and evaluation plots.

### Backtest / backtesting

A particular kind of evaluation: testing a model on historical data by repeatedly pretending to stand at different points in the past, making predictions from there, and comparing them with what actually happened. Also called retrospective validation. Nothing in a backtest may see data beyond the pretended now-point.

### Hindcast

A prediction made for a past period as if in real time, using only what would have been known at the pretended now-point. A backtest produces hindcasts and scores them. Also called a retrospective forecast or reforecast. In the climate literature, "hindcast" mostly means retrospective climate forecasts from forecasting centres.

### Pretended now-point

The point in the past a backtest pretends to stand at. Data after it is withheld. The forecasting literature calls it the forecast origin. The code calls it a split period (`split_periods` on `Backtest`).

### Rolling-origin evaluation

Also called time-series cross-validation. Repeating the backtest from many successive pretended now-points and pooling the results, which gives a fuller picture of operational performance than a single split. In `BacktestParams`, `n_splits` sets how many now-points are used and `stride` how many periods apart they are. chap-core uses an expanding window: the historical data the model is given grows with each now-point. The model is trained once unless `n_retrain` is raised.


