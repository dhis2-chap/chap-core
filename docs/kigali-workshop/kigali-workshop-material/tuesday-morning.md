# Tuesday Morning - 24 Feb

You could have a look at your model of main interest among several open-source models already integrated with Chap:
- An Exponential Smoothing Model, implemented in R: https://github.com/chap-models/auto_ets
- An ARIMA model, implemented in R: https://github.com/chap-models/auto_arima
- An Hierarchical Bayesian model with INLA-R based model training (R): https://github.com/chap-models/chap_auto_ewars
- A regression-based model, implemented in Python: https://github.com/knutdrand/simple_multistep_model
- If you want to look at a very simple time series baseline, this just predicts the historic mean (R code): https://github.com/chap-models/mean

Most of these models have a file called train.R (or train.py) that trains a model based on historic data, 
and then a file called predict.R (or predict.py) that forecasts disease cases ahead in time based on the model.
Note, though, that some models always trains the model at the time of prediction, meaning that for some models train and predict will then both be inside the predict.R file.   