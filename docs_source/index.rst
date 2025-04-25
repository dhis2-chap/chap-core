The Chap Modeling Platform
==============================================================

The Chap Modeling Platform is a platform for forecasting and for assessing forecasts of climate-sensitive health outcomes.
In the early phase, the focus is on vector-borne diseases like malaria and dengue.

The platform can perform data parsing, data integration, forecasting based on any of
multiple supported models, automatic brokering of compatible models for a given prediction context and robust forecast assessment and method comparison.

This documentation contains technical information about installing and using the modelling platform. For more general information about the Chap project,
 we refer to `the CHAP project wiki <https://github.com/dhis2-chap/chap-core/wiki>`_. 

----

All pages
----------

The following is an overview of all pages in the documentation:

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Installation and getting started

   installation/installation
   installation/modeling-app-setup
   changelog


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using the Modeling App with the Chap modelling platform and DHIS2

   modeling-app/*


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using as a CLI Tool

   chap-cli/*


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using Chap modelling platform as a Python library

   python-api/*


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Integrating external or custom models with the Chap modelling platform

   external_models/making_external_models_compatible
   external_models/developing_custom_models
   external_models/external_model_specification
   external_models/running_external_models
   external_models/integrating_external_models_with_dhis2
   external_models/wrapping_gluonts
   external_models/overview_of_supported_models
   external_models/data_requirements_in_chap
   external_models/running_models_with_examples


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contributor guide 

   contributor/getting_started
   contributor/windows_contributors
   contributor/code_overview
   contributor/vocabulary
   contributor/testing
   contributor/writing_building_documentation
   contributor/r_docker_image
