Climate health assessment platform (CHAP)
==========================================

CHAP is a platform for forecasting and for assessing forecasts of climate-sensitive health outcomes.
In the early phase, the focus is on vector-borne diseases like malaria and dengue
The platforms is to perform data parsing, data integration, forecasting based on any of multiple supported models, automatic brokering of compatible models for a given prediction context and robust forecast assessment and method comparison.

The current version has basic data handling functionality in place, and is almost at a stage where it supports running a first external model (EWARS-Plus)


.. toctree::
   :maxdepth: 0
   :caption: Contents:

   installation
   external_models
   tutorials/wrapping_gluonts
   tutorials/downloaded_json_data


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


