# Planned and ongoing functionality and resources for modellers in Chap
The Chap modelling platform brings together a broad range of functionality for streamlining climate health modelling into a unified ecosystem.
This document provides an overview of ongoing and planned functionality and features, mainly meant for model developers and Chap developers.
For functionality that is already available, see [capabilities.md](capabilities.md).

### At the core of Chap is the plugin-like support for incorporating models into the platform:
* There is ongoing work to change from the current minimalist examples to a more sophisticated starting point based on an SDK in [Python]() and [R]().
* There is also ongoing work to change from the current scheme of each model being run as a subprocess (possibly within a Docker container) to instead having the models provide a REST API for communication with the Chap Platform (the [chapkit project](https://github.com/dhis2-chap/chapkit)

### Data input:
* There are also future plans for a [collection of metadata for restricted data]() potentially available from specific countries (TODO: and more?).
* There is ongoing work on [generation of synthetic climate and health datasets]() for understanding model behaviour and stress-testing model in particular settings.
* There is ongoing work on supporting the [computation of endemic channels]() (outbreak thresholds), as well as functionality to [compute outbreak periods]() (binary representation of early warning forecast) based on outbreak threshold and probabilistic disease forecasts.

### Ways of running models:
* Ongoing work on SDKs will bring streamlined ways of running models in their native programming language.
* There is ongoing work on streamlined REST api setup for methods, allowing any model to also be run through a REST api (including over the internet??)

### Rigorous evaluation of model predictions:
* There is ongoing work on a [benchmarking server]() that streamlines extensive evaluation and continuous evaluation through development.
* There are future plans for [federated model evaluation]() through Chap, in which a model can be evaluated on data across multiple countries without needing to be provided access to the data itself for these countries
* Plans for a standard benchmark setup that allows any model integrated with Chap to be assessed on a standard collection of data using a standard collection of metrics and visualisations

### Chap's optional streamlined setup of [connection to DHIS]() has the following planned additional features:
* Interoperability with the full set of [DHIS2 ecosystem tools and functionalities](), including planned support for missing data analysis and imputation, for endemic threshold definition and outbreak inference, for derived variable computation and dashboard visualisation of prediction.

### Beyond the core features described above, the platform also plans to offer the following features to any model integrated with it
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

### Complementary features in the DHIS2 ecosystem (outside but useful for Chap and modelling)
- In order to compute outbreak probabilities, the predicted number of cases need to be compared to an outbreak threshold (often per region and month), also referred to as endemic channel. We are currently developing a flexible scheme for computing and using such thresholds to derive forecasts of outbreak probabilties.

### Research
* We have many ambitions on [research]() and scientific publications on technical, IS and climate health aspects of Chap

### Documentation, tutorials and capacity development
* See [capabilities.md](capabilities.md) for currently available documentation resources.

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
