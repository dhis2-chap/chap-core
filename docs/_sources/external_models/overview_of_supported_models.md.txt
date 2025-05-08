# Overview of supported models
## Autoregressive weekly
This model has been developed internally by the Chap team. Autoregressive weekly is a deep learning model based on DeepAR that uses rainfall and temperature as climate predictors and models disease counts with a negative binomial distribution where the parameters are estimated by a recurrent neural network. The current model available through Chap can be [found here](https://github.com/knutdrand/weekly_ar_model).

## Epidemiar
Epidemiar is a Generalizes additive model (GAM) used for climate health forecasts. It requires weekly epidemilogical data, like disease cases and population, and daily enviromental data. As most of the data in CHAP is monthly or weekly we pass weakly data to the model, and then naively expand weekly data to daily data, which the epidemiar library again aggregates back to weekly data. The model produces a sample for each location per time point with and upper and lower boundary for some unknown quantiles. For more information regarding the model look [here](https://github.com/dhis2-chap/epidemiar_example_model). An example of running the model from the CHAP command line interface is
```
chap evaluate --model-name https://github.com/dhis2-chap/epidemiar_example_model --dataset-csv LOCAL_FILE_PATH/laos_test_data.csv --report-filename report.pdf --debug --n-splits=3
```
which requires that `laos_test_data` is saved locally, but we are working on making weekly datasets available internaly in CHAP.

## EWARS
EWARS is a Bayesian hierarchical model implemented with the INLA library. We use a negative binomial likelihood in the observation layer and combine several latent effect, both spatial and temporal, in the latent layer. The latent layer is log-transformed and scaled by the population, so it effectivaly models the proprotion of cases in each region. Specifically the latent layers combine a first order cyclic random walk to capture the seasonal effect, this is also included in the lagged exogenous variables rainfall and temperature, then a spatial smoothing with an ICAR and an iid effect to capture the spatial heterogeneity. The ICAR and iid can also be combined, scaled and reparameterized to the BYM2 model. Further information is available in the [model repository](https://github.com/dhis2-chap/chap_auto_ewars). An example of running the model from the CHAP command line interface is
```
chap evaluate --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=3
```

## ARIMA
A general ARIMA model is a timeseries model with an autoregressive part, a moving average part and the option to difference the original timeseries, often to make it stationary. Additonally we have lagged rainfall and temperature, which actually makes this an ARIMAX model, where the X indicates exogenous variables. This model handles each region individually and it expects monthly data for all the covariates. The model utilizes the `arima` function which chooses the order of the differencing, autoregression and the moving average for us. Further information is available in the [model repository](https://github.com/dhis2-chap/Madagascar_ARIMA). An example of running the model from the CHAP command line interface is
```
chap evaluate --model-name https://github.com/dhis2-chap/Madagascar_ARIMA --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=3
```
