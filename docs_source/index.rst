Climate Health Analytics Platform (CHAP)
==============================================================

CHAP is a platform for forecasting and for assessing forecasts of climate-sensitive health outcomes.
In the early phase, the focus is on vector-borne diseases like malaria and dengue.

The platform can perform data parsing, data integration, forecasting based on any of
multiple supported models, automatic brokering of compatible models for a given prediction context and robust forecast assessment and method comparison.

This documentation contains technical information about installing and using CHAP. For more general information about the project, we refer to `the CHAP project wiki <https://github.com/dhis2-chap/chap-core/wiki>`_. 

----

All pages
----------

The following is an overview of all pages in the documentation:

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation and getting started

   installation
   installation/*
   changelog


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using the Predictions App with CHAP (and integration with DHIS2)

   prediction-app/*
   tutorials/downloaded_json_data


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using CHAP as a library

   api_docs
   tutorials/wrapping_gluonts


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using the CHAP Core REST-API

   rest-api/*


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Integrating external or custom models with CHAP

   external_models/making_external_models_compatible.rst
   external_models/developing_custom_models.rst
   external_models/external_model_specification.rst
   external_models/running_external_models.rst
   external_models/integrating_external_models_with_dhis2.rst


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Development guide 

   developer/getting_started
   developer/code_overview
   developer/testing
   developer/writing_building_documentation
   developer/code_guidelines
   developer/manual_runs
