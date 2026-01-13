# Current and planned functionality and resources for modellers in Chap
The Chap modelling platform brings together a broad range of functionality for streamlining climate health modelling into a unified ecosystem.
This document provides an overview of existing and planned functionality and features, mainly meant for model developers and Chap developers.

### At the core of Chap is the plugin-like support for incorporating models into the platform: 
* This is based on a scheme where a model provides functionality to train a model and predict ahead in time, defining its entry points in an MLFlow-based format.
* Any model adhering to this can be used in the platform by having the model available as a github repository and providing the url for this repo to the Chap platform. 
* Chap can run the model either in its own python environment or in a docker container (where the model points to a docker image that it can be run within).
* Model developers are offered a template (in [Python](https://github.com/dhis2-chap/chap_model_template) or [R](https://github.com/dhis2-chap/chap_model_template_r)) or minimalist example of a working model codebase to start from (in [Python](https://github.com/dhis2-chap/minimalist_example) or [R](https://github.com/dhis2-chap/minimalist_example_r)).
* There is ongoing work to change from the current minimalist examples to a more sophisticated starting point based on an SDK in [Python]() and [R](). 
* There is also ongoing work to change from the current scheme of each model being run as a subprocess (possibly within a Docker container) to instead having the models provide a REST API for communication with the Chap Platform (the [chapkit project](https://github.com/dhis2-chap/chapkit)  

Integrating with the Chap platform allows to focus only on the model itself, and by having it adhere to our standard interface the model can rely on the platform for the central aspects of data input, ways of running models, model evaluation and optional DHIS2 integration.

### Data input: 
* Chap allows input from a [well-defined csv format]() for harmonised climate and health data.
* A broad repository of [public harmonised climate and health data]() is available in this format and can be directly used with a model. 
* There are also future plans for a [collection of metadata for restricted data]() potentially available from specific countries (TODO: and more?).
* There is ongoing work on [generation of synthetic climate and health datasets]() for understanding model behaviour and stress-testing model in particular settings. 
* There is ongoing work on supporting the [computation of endemic channels]() (outbreak thresholds), as well as functionality to [compute outbreak periods]() (binary representation of early warning forecast) based on outbreak threshold and probabilistic disease forecasts.  
### Ways of running models:
* Any model can always be run through its native programming language, and the ongoing work on SDKs will bring streamlined ways of running
* Any model, implemented in any language, can be run through the [Chap command-line interface]()
* There is ongoing work on streamlined REST api setup for methods, allowing any model to also be run through a REST api (including over the internet??)
* As described below, through optional streamlined DHIS2 connection, a model can be run through a GUI using the Modelling App   

### Rigorous evaluation of model predictions: 
* Model predictions can be contrasted to truth according to our precisely defined (TODO) [evaluation scheme]() that follows our dogma of what constitutes [appropriate evaluation]().
* There is ongoing work on a [benchmarking server]() that streamlines extensive evaluation and continuous evaluation through development.
* There are future plans for [federated model evaluation]() through Chap, in which a model can be evaluated on data across multiple countries without needing to be provided access to the data itself for these countries
* Plans for a standard benchmark setup that allows any model integrated with Chap to be assessed on a standard collection of data using a standard collection of metrics and visualisations  

### Chap further includes optional streamlined setup of [connection to DHIS](), which provides the following additional features:
* Direct data input from DHIS2, which through the [Climate App]() and [Climate Tools]() may contain up-to-date, harmonised climate and health data according to well-defined criteria.
* Direct dissemination of predictions back to DHIS2
* Using/offering the Modelling App as a GUI for your own and reference models: Configuring, tuning, training, evaluating and predicting with models, as well as visualising data, model predictions and evaluations.   
* Interoperability with the full set of [DHIS2 ecosystem tools and functionalities](), including planned support for missing data analysis and imputation, for endemic threshold definition and outbreak inference, for derived variable computation and dashboard visualisation of prediction.

* In addition to the plugin-like system for models, we similarly offer:
* A plugin-like system for evaluation metrics, allowing anyone to [contribute implementations of custom metrics]() (formulas) for evaluating model predictions against truth   
* A plugin-like system for visualisations, allowing anyone to contribute [visualisations of data]() or [visualisations for model evaluation]().  

### Beyond the core features described above, the platform also currently or in the future offers the following features to any model integrated with it
* Persistency: Both trained models and their predictions on different datasets can be stored according to our [persistency support](), allowing to run trained models on new data and set up comparative evaluations. 
* AutoML: There is ongoing work to support [automatic model tuning]() (model Hyper-parameters to be tuned), as well as planned work to allow automatic variable selection and automatic selection of model to use on a given dataset.
* Ensemble model learning: There is ongoing work on combining multiple models to get both mean predictions and uncertainty combined across models, ranging from: 
  * Ongoing work on basic aggregation and manual or automatically learned weighting of multiple models 
  * Plans for an adaptive ensemble approach (mixture of experts), where model choice within an ensemble is dynamically set based on data
* Model introspection and explainability:
	* Ongoing work on an open way for models to provide any [information on a trained model or model predictions]() 
    * Ongoing work on a generic ontology and [protocol for models to communicate model properties]() (like variable importance) in a way that can be easily compared across models
	* Planned work on a generic data perturbation scheme to infer model (across-model comparable) characteristics (like variable importance) from the platform side through the standard train and prediction endpoints (i.e. without models having to implement anything related to model introspection/explainability)   
	* Planned work on missing data sensitivity analysis (by randomly dropping data and assessing its effect)
    * Planned work on automatic brokering of compatible models for a given prediction context according to metadata (filtering models based on chosen data availability and decision need)
* Plans for overall summary of forecasting analyses, including details of data, training and prediction skill

### Research
* We have many ambitions on [research]() and scientific publications on technical, IS and climate health aspects of Chap  

### Documentation, tutorials and capacity development 
* We provide an [overall Chap documentation](), with subparts for 
  * How to learn about and [integrate with Chap as modeller]()  
  * How to [contribute to the core Chap codebase](). 
* We provide capacity building material on [learning modelling based on Chap]()
* We have a separate [tutorial](https://github.com/norajeanett/Chap-core-Guidelines/blob/main/README.md) meant for master students or similar to get started with Chap. 

### Collaboration and supervision
* The following PhD students do their PhD project with Chap being central:
  * Herman Tretteteig
  * Halvard Emil Sand-Larsen
* The following Master students are currently working on concrete aspects of Chap:
  * Lilu Zhan: autoML ([tutorial]())
  * Nora Jeanett Tønnessen: modularised visualisation ([tutorial]())
  * Markus Byrkjedal Slyngstad: model cards ([tutorial]())
  * Behdad Nikkhah: ensemble learning ([tutorial]())
  * Hamed Hussaini: model introspection ([tutorial]())
  * Ali Hassan: federated model evaluation ([tutorial]())
  * Andre Maharaj Gregussen: modularised evaluation metric definition ([tutorial](https://github.com/AndreGregu/Assessment_example_chap_compatible/blob/andre_new_code/TUTORIAL.md))
* The following Master students are planned to contribute to Chap in the time ahead:
  * Leander S. Parton, Hans Andersen, August Aspelien, William Henrik Behn, Ole Martin Skovly  Henning, Sigurd Smeby, Aulona Sylanaj
* The following Master students have delivered a thesis connected to Chap in the past:
  * Martin Hansen Bolle and Ingar Andre Benonisen: synthetic datasets
* The following external collaborators have contributed to Chap:
  * Harsha Halgamuwe Hewage: The use of Chap for drug logistics planning 