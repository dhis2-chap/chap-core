# Tuesday Morning - 24 Feb

An introduction to time series modeling, as well as more specifically into supply chain forecasting (slides), given by Bahman Rostami-Tabar and Harsha Halgamuwe Hewage:
- [Classical time series models](tuesday-morning-files/classical_rwanda.pdf)
- [Forecasting in Supply Chain](tuesday-morning-files/supply_chain.pdf)

A set of R Markdown documents (and soon webinar recordings) is also available for Bayesian modeling according to a process of data exploration followed by manual model selection, given by Ania Kawiecki Peralta and Carles Milà of the Barcelona Supercomputing Center:
- [Model selection workflow using GHRmodel](https://gitlab.earth.bsc.es/ghr/ghr4dhis2/-/tree/selection/)

As part of this session, one can look at a model of main interest among several open-source models already integrated with Chap:
- An Exponential Smoothing Model, implemented in R: [auto_ets](https://github.com/chap-models/auto_ets)
- An ARIMA model, implemented in R: [auto_arima](https://github.com/chap-models/auto_arima)
- An Hierarchical Bayesian model with INLA-R based model training (R): [chap_auto_ewars](https://github.com/chap-models/chap_auto_ewars)
- A regression-based model, implemented in Python: [simple_multistep_model](https://github.com/knutdrand/simple_multistep_model)
- If you want to look at a very simple time series baseline, this just predicts the historic mean (R code): [mean](https://github.com/chap-models/mean)

Most of these models have a file called train.R (or train.py) that trains a model based on historic data,
and then a file called predict.R (or predict.py) that forecasts disease cases ahead in time based on the model.
Note, though, that some models always trains the model at the time of prediction, meaning that for some models train and predict will then both be inside the predict.R file.
