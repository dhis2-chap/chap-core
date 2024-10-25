Climate Health Analytics Platform (CHAP)
==============================================================
Installing and using CHAP
"""""""""""""""""""""""""""""""""""""""""""""

CHAP is a platform for forecasting and for assessing forecasts of climate-sensitive health outcomes.
In the early phase, the focus is on vector-borne diseases like malaria and dengue
The platforms is to perform data parsing, data integration, forecasting based on any of 
multiple supported models, automatic brokering of compatible models for a given prediction context and robust forecast assessment and method comparison.

This page container documentation regarding installing and using CHAP. This contain information about every CHAP product, such as CHAP Core, Prediction App, and Climate App.


The current version has basic data handling functionality in place, and is almost at a stage where it supports running a first external model (EWARS-Plus)


.. toctree::
   :maxdepth: 0
   :caption: Contents:
   
   prediction-app/prediction-app
   docker-compose-doc
   installation
   external_models
   developing_custom_models
   tutorials/wrapping_gluonts

API documentation
===================

Data Fetching
--------------

Functionality for fetching data


.. currentmodule:: climate_health.fetch

.. autofunction:: gee_era5

.. currentmodule:: climate_health.data

.. autoclass:: DataSet
    :members: from_period_observations, from_pandas, to_pandas

.. autoclass:: PeriodObservation


