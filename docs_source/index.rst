Climate Health Analytics Platform (CHAP)
==============================================================

CHAP is a platform for forecasting and for assessing forecasts of climate-sensitive health outcomes.
In the early phase, the focus is on vector-borne diseases like malaria and dengue
The platform can perform data parsing, data integration, forecasting based on any of
multiple supported models, automatic brokering of compatible models for a given prediction context and robust forecast assessment and method comparison.

The current version has basic data handling functionality in place, and is almost at a stage where it supports running a first external model (EWARS-Plus)

This documentation contains information about installing and using CHAP. The documentation is divided into sections depending on use-case:

- For users who want to fully install CHAP locally or on a server, e.g. to integrate with DHIS2, we recommend :doc:`setting up CHAP with docker compose <docker-compose-doc>` and :doc:`using the Prediction app <prediction-app/prediction-app>`.
- For users who want to develop custom forecasting models and run or benchmark these through CHAP, or to simply evaluate external models on small example datasets, we recommend :ref:`installing the chap-core Python package <installation>` and folowing the guides on :ref:`integrating external models <external_models>` and :ref:`developing custom models <developing_custom_models>`.
- For users who want to use CHAP as a library, we refer to the tutorials and API documentation (see the menu).


All pages
----------

The following is an overview of all pages in the documentation:

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Installation and getting started

   installation
   docker-compose-doc

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using the Predictions App with CHAP (and integration with DHIS2)

   prediction-app/*
   tutorials/downloaded_json_data


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Integrating external or custom models with CHAP

   external_models/*


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using CHAP as a library

   tutorials/wrapping_gluonts
   api_docs


