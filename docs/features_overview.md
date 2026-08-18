# Current and planned functionality and resources for modellers in Chap
The Chap modelling platform brings together a broad range of functionality for streamlining climate health modelling into a unified ecosystem.
This document provides an overview of existing and planned functionality and features, mainly meant for model developers and Chap developers.

### At the core of Chap is the plugin-like support for incorporating models into the platform:
* Models must implement a train and predict interface. Any model adhering to this can be used in the platform by having the model available as a GitHub repository and providing the URL for this repo to the Chap platform.
* Model developers are offered templates in [Python](https://github.com/dhis2-chap/chap_model_template) and [R](https://github.com/dhis2-chap/chap_model_template_r), as well as minimalist example codebases in [Python](https://github.com/dhis2-chap/minimalist_example) and [R](https://github.com/dhis2-chap/minimalist_example_r).
* Chap can run models in several execution environments: Python (`uv`), R (`renv`), Conda, MLFlow projects, or a Docker container (where the model points to a Docker image).
* Models can also expose themselves as HTTP services, in which case Chap communicates with them over a REST API via the [CHAPKit project](https://github.com/dhis2-chap/chapkit). This is supported through the v2 service registry API.

Integrating with the Chap platform allows a model developer to focus only on the model itself. By having the model adhere to our standard interface, the model can rely on the platform for data input, ways of running models, model evaluation, and optional DHIS2 integration.

### Data input:
* Chap accepts a well-defined CSV format for harmonised climate and health data.
* A broad repository of public harmonised climate and health data is available and can be directly used with a model.
* Spatial data (polygons) can be provided as GeoJSON and is integrated with the health data.
* There is ongoing work on [generation of synthetic climate and health datasets]() for understanding model behaviour and stress-testing models in particular settings.
* There is ongoing work on supporting the computation of endemic channels (outbreak thresholds), as well as functionality to compute outbreak periods (binary representation of early warning forecast) based on outbreak thresholds and probabilistic disease forecasts.

### Ways of running models:
* Any model can always be run through its native programming language (Python, R, etc.).
* Any model, implemented in any language, can be run through the [Chap command-line interface]().
* A REST API (FastAPI-based) is available for running models programmatically, including async job execution backed by Celery and Redis.
* As described below, through optional streamlined DHIS2 connection, a model can be run through a GUI using the Modelling App.

### Rigorous evaluation of model predictions:
* Model predictions can be contrasted to truth according to our precisely defined [evaluation scheme]() that follows our dogma of what constitutes appropriate evaluation.
* Evaluation is performed through a backtesting framework that supports multi-period backtesting with configurable strides and prediction horizons.
* Results are stored in xarray/NetCDF format for downstream analysis.
* There are future plans for [federated model evaluation]() through Chap, in which a model can be evaluated on data across multiple countries without needing to be provided access to the data itself.
* Plans for a standard benchmark setup that allows any model integrated with Chap to be assessed on a standard collection of data using a standard collection of metrics and visualisations.

### Chap further includes optional streamlined setup of connection to DHIS2, which provides the following additional features:
* Direct data input from DHIS2, which through the [Climate App]() and [Climate Tools]() may contain up-to-date, harmonised climate and health data according to well-defined criteria.
* Direct dissemination of predictions back to DHIS2.
* Using/offering the Modelling App as a GUI for your own and reference models: Configuring, tuning, training, evaluating and predicting with models, as well as visualising data, model predictions and evaluations.
* Interoperability with the full set of [DHIS2 ecosystem tools and functionalities](), including planned support for missing data analysis and imputation, endemic threshold definition, outbreak inference, derived variable computation, and dashboard visualisation of predictions.

### In addition to the plugin-like system for models, we similarly offer:
* A plugin-like system for evaluation metrics, allowing anyone to [contribute implementations of custom metrics](contributor/creating_custom_metrics.md) (formulas) for evaluating model predictions against truth. Currently implemented metrics include MAE, RMSE, MAPE, CRPS (and log1p variant), Winkler Score, Percentile Coverage, Outbreak Detection metrics (sensitivity, specificity, accuracy), Peak Diff, Peak Period Lag, and Above Truth Ratio.
* A plugin-like system for backtest visualisations, allowing anyone to contribute visualisations for model evaluation. Several visualisation types are available including horizon-location grid plots, predicted vs actual plots, and sample bias visualisations.
* A plugin-like system for metric visualisations, allowing anyone to contribute visualisations of computed metrics over time, space, or horizon. Built-in types include horizon mean/sum, time period mean/sum, regional distribution, and geospatial metric maps.
* A plugin-like system for endemic channel (threshold) strategies, with a seasonal threshold implementation available.

All four plugin systems are based on decorator-based registration (`@metric()`, `@backtest_plot()`, `@metric_plot()`, `@threshold()`) and are extensible by anyone.

### Beyond the core features described above, the platform also currently or in the future offers the following features to any model integrated with it:
* Persistency: Both trained models and their predictions on different datasets can be stored using our SQLAlchemy-based persistence layer (with Alembic migrations), allowing trained models to be run on new data and enabling comparative evaluations.
* Extended prediction horizons: Any model can be wrapped with `ExtendedPredictor` to make predictions beyond its maximum prediction length through iterative prediction.
* Hyperparameter optimisation (HPO): The platform supports automatic tuning of model hyperparameters through a configurable search interface, currently with a random search implementation. This can be triggered through the `evaluate` CLI command.
* Preference learning: An A/B testing framework allows interactive comparison of model candidates based on visual inspection or metric-based decision modes.
* Model introspection and explainability:
    * LIME-based local interpretability is implemented, allowing perturbation analysis and surrogate model generation to understand model behaviour.
    * Planned work on a generic ontology and protocol for models to communicate model properties (like variable importance) in a way that can be compared across models.
    * Planned work on a generic data perturbation scheme to infer model characteristics from the platform side through the standard train and prediction endpoints (without models having to implement anything related to explainability).
* Model cards: Automated model documentation can be generated via the `generate_modelcard` CLI command.
* Counterfactual/causal analysis: A `causal` CLI command allows comparative impact analysis across different scenarios.
* Ensemble model learning: Support for combining multiple models is planned but not yet implemented.
* AutoML: Hyperparameter optimisation is implemented. Automatic variable selection and automatic model selection for a given dataset are planned.
* Plans for an overall summary of forecasting analyses, including details of data, training, and prediction skill.

### Complementary features in the DHIS2 ecosystem (outside but useful for Chap and modelling)
* In order to compute outbreak probabilities, the predicted number of cases need to be compared to an outbreak threshold (often per region and month), also referred to as endemic channel. We are currently developing a flexible scheme for computing and using such thresholds to derive forecasts of outbreak probabilities.

### Research
* We have many ambitions on [research]() and scientific publications on technical, IS and climate health aspects of Chap.

### Documentation, tutorials and capacity development
* We provide an [overall Chap documentation](), with subparts for:
  * How to learn about and [integrate with Chap as a modeller]()
  * How to [contribute to the core Chap codebase]()
* We provide capacity building material on [learning modelling based on Chap]().
* We have a separate [tutorial](https://github.com/norajeanett/Chap-core-Guidelines/blob/main/README.md) meant for master students or similar to get started with Chap.
