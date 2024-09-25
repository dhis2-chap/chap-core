# Climate Health Analysis Platform (CHAP)
CHAP offers a platform for analysing the relationship between climate and health. The platform is designed to be modular and flexible, allowing for easy integration of new models and data sources. The platform is designed to be used by researchers and public health professionals to forecast and assess the impact of climate on health outcomes.

# Installation

Basic installation can be done using pip:

    $ pip install git+https://github.com/dhis2/chap-core.git

If running models, you may also need to install:

- Docker, if running a model that runs through docker
- [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation), if running a model that uses Python virtual environments

# Documentation

The main documentation is located at [https://dhis2.github.io/chap-core/](https://dhis2.github.io/chap-core/).

# Usage

The following shows basic usage of the platform. Follow the link to the documentation above for more details.

## Evaluate a public model on public data
CHAP supports evaluating models that are defined using the MLflow specification for machine learning models (link coming). Such models can e.g. exist in Github repositories. CHAP also has some built-in example data that can be used to evaluate models. The following example shows how to evaluate an Ewars model located on Github ([https://github.com/sandvelab/chap_auto_ewars](https://github.com/sandvelab/chap_auto_ewars))  using the ISMIP dataset:

```bash
chap evaluate --model-name https://github.com/sandvelab/chap_auto_ewars --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil
```

The above example requires that you have installed chap with pip and also that you have Docker available.


## Fetching Polygon Data
Fetch polygons for regions of interest in a country (on admin1 level). The following example fetches polygons for two regions in Norway

```python
    import chap_core.fetch

polygons = chap_core.fetch.get_area_polygons('Norway', ['Oslo', 'Akershus'])
assert [feature.id for feature in polygons.features] == ['Oslo', 'Akershus']
```
Region names that are not recognized are skipped:

```python
    polygons = climate_health.fetch.get_area_polygons('Norway', ['Oslo', 'Akershus', 'Unknown'])
    assert [feature.id for feature in polygons.features] == ['Oslo', 'Akershus']
```

## Fetching climate data
Fetching climate data through Google Earth Engine. The following example fetches temperature data from the ERA5 dataset for the regions of interest. You need to have an an account at gee, and pass the credentials to the method to make this work: https://developers.google.com/earth-engine/guides/auth

```python

import chap_core.fetch

credentials = dict(account='demoaccount@demo.gserviceaccount.com', private_key='private_key')
polygons = open("polygon_file.geojson").read()
start_period = '202001'  # January 2020
end_period = '202011'  # December 2020
band_names = ['temperature_2m', 'total_precipitation_sum']
data = chap_core.fetch.gee_era5(credentials, polygons, start_period, end_period, band_names)
print(data)
start_week = '2020W03'  # Week 3 of 2020
end_week = '2020W05'  # Week 5 of 2020
data = chap_core.fetch.gee_era5(credentials, polygons, start_week, end_week, band_names)
print(data)
```
