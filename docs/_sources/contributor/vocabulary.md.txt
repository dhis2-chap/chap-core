# Vocabulary (domain specific terms used in the code)


### Model template

A model template is a flexible "model" which can be configured. A model template typically presents various options (hyperparameters, possible covariates, etc) which are open for configuration. 


### Configured model

A configured model can be made from a model template by applying chocies to the options presented by the model template. Only a configured model can actually be trained on a given dataset (as opposed to a model template, since a model template does not necessarily have enough information about how to train or predict).

### ExternalModel

ExternalModel is a wrapper around an external model (that can e.g. be an R model) to make it compatible with the interface of ConfiguredModel. This means that ExternalModel has train/predict similarily to ConfiguredModel, but these methods are wrappers that runs the train/predict of external models.

## Some other terms we use

Backtest: Is the same as evaluation for now (used as a term in the REST API)


## Runner

A runner is something that can run commands. ExternalModels (not ConfiguredModels) have a Runner object attached to them. When train/predict is called, the runner is handling how to j6o05..,m.
