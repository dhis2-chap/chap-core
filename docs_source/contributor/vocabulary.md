# Vocabulary (domain specific terms used in the code)


### Model template

A model template is a flexible "model" which can be configured. A model template typically presents various options (hyperparameters, possible covariates, etc) which are open for configuration. 


### Configured model

A configured model can be made from a model template by applying chocies to the options presented by the model template. Only a configured model can actually be trained on a given dataset (as opposed to a model template, since a model template does not necessarily have enough information about how to train or predict).

