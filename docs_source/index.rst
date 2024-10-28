Climate Health Analytics Platform (CHAP)
==============================================================

CHAP is a platform for forecasting and for assessing forecasts of climate-sensitive health outcomes.
In the early phase, the focus is on vector-borne diseases like malaria and dengue
The platform can perform data parsing, data integration, forecasting based on any of
multiple supported models, automatic brokering of compatible models for a given prediction context and robust forecast assessment and method comparison.

The current version has basic data handling functionality in place, and is almost at a stage where it supports running a first external model (EWARS-Plus)

This documentation contains information about installing and using CHAP. The documentation is divided into sections depending on use-case:

- For users who want to test out a full installation of CHAP locally, we recommend following the guide on :doc:`settiing up CHAP with docker compose <docker-compose-doc>`.
- For users who want to develop custom forecasting models and run or benchmark these through CHAP, we recommend installing the chap-core Python package and folowing the guides on :ref:`integrating external models <external_models>` and :ref:`developing custom models <developing_custom_models>`.


The following is an overview of all pages in the documentation:



.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Installation and getting started

   installation

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Setting up CHAP with docker and integration with DHIS2

   prediction-app/*
   docker-compose-doc
   tutorials/downloaded_json_data


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Integration external or custom models with CHAP

   external_models/*


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Using CHAP as a library

   tutorials/wrapping_gluonts
   api_docs


