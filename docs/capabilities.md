# Current capabilities of Chap
The Chap modelling platform brings together a broad range of functionality for streamlining climate health modelling into a unified ecosystem.
This document provides an overview of functionality and features that are already available today, mainly meant for model developers and Chap developers.
For ongoing and planned work, see [features_overview.md](features_overview.md).

### At the core of Chap is the plugin-like support for incorporating models into the platform:
* This is based on a scheme where a model provides functionality to train a model and predict ahead in time, defining its entry points in an MLFlow-based format.
* Any model adhering to this can be used in the platform by having the model available as a github repository and providing the url for this repo to the Chap platform.
* Chap can run the model either in its own python environment or in a docker container (where the model points to a docker image that it can be run within).
* Model developers are offered a template (in [Python](https://github.com/dhis2-chap/chap_model_template) or [R](https://github.com/dhis2-chap/chap_model_template_r)) or minimalist example of a working model codebase to start from (in [Python](https://github.com/dhis2-chap/minimalist_example) or [R](https://github.com/dhis2-chap/minimalist_example_r)).

Integrating with the Chap platform allows to focus only on the model itself, and by having it adhere to our standard interface the model can rely on the platform for the central aspects of data input, ways of running models, model evaluation and optional DHIS2 integration.

### Data input:
* Chap allows input from a [well-defined csv format]() for harmonised climate and health data.
* A broad repository of [public harmonised climate and health data]() is available in this format and can be directly used with a model.

### Ways of running models:
* Any model can always be run through its native programming language.
* Any model, implemented in any language, can be run through the [Chap command-line interface]()
* Through optional streamlined DHIS2 connection, a model can be run through a GUI using the Modelling App

### Rigorous evaluation of model predictions:
* Model predictions can be contrasted to truth according to our precisely defined (TODO) [evaluation scheme]() that follows our dogma of what constitutes [appropriate evaluation]().

### Chap further includes optional streamlined setup of [connection to DHIS](), which provides the following additional features:
* Direct data input from DHIS2, which through the [Climate App]() and [Climate Tools]() may contain up-to-date, harmonised climate and health data according to well-defined criteria.
* Direct dissemination of predictions back to DHIS2
* Using/offering the Modelling App as a GUI for your own and reference models: Configuring, tuning, training, evaluating and predicting with models, as well as visualising data, model predictions and evaluations.
* Interoperability with the full set of [DHIS2 ecosystem tools and functionalities]().

In addition to the plugin-like system for models, we similarly offer:
* A plugin-like system for evaluation metrics, allowing anyone to [contribute implementations of custom metrics](contributor/creating_custom_metrics.md) (formulas) for evaluating model predictions against truth
* A plugin-like system for visualisations, allowing anyone to contribute [visualisations of data]() or [visualisations for model evaluation]().

### Beyond the core features described above, the platform also currently offers the following features to any model integrated with it
* Persistency: Both trained models and their predictions on different datasets can be stored according to our [persistency support](), allowing to run trained models on new data and set up comparative evaluations.
* Extended prediction horizons: Any model can be wrapped with ExtendedPredictor to make predictions beyond its maximum prediction length through iterative prediction.

### Complementary features in the DHIS2 ecosystem (outside but useful for Chap and modelling)
- A script is available that can be used to populate outbreak threshold values into DHIS2, which can then be imported to and used with Chap.

### Documentation, tutorials and capacity development
* We provide an [overall Chap documentation](), with subparts for
  * How to learn about and [integrate with Chap as modeller]()
  * How to [contribute to the core Chap codebase]().
* We provide capacity building material on [learning modelling based on Chap]()
* We have a separate [tutorial](https://github.com/norajeanett/Chap-core-Guidelines/blob/main/README.md) meant for master students or similar to get started with Chap.
