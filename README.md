# Climate Health Anlysis Platform (CHAP)
CHAP offers a platform for analysing the relationship between climate and health. The platform is designed to be modular and flexible, allowing for easy integration of new models and data sources. The platform is designed to be used by researchers and public health professionals to forecast and assess the impact of climate on health outcomes.

# Installation

    $ pip install git+https://github.com/dhis2/chap-core.git

# Usage
## Fetching climate data
Fetching climate data through Google Earth Engine. The following example fetches temperature data from the ERA5 dataset for the regions of interest.


```python

    import climate_health.fetch
    credentials = dict(account='demoaccount@demo.gserviceaccount.com', private_key='private_key')
    polygons = open("../example_data/Organisation units.geojson").read()
    start_period = '202001' # January 2020
    end_period = '202011' # December 2020
    band_names = ['temperature_2m', 'total_precipitation_sum']
    data = climate_health.fetch.gee_era5(credentials, polygons, start_period, end_period, band_names)
    print(data)
    start_week = '2020W03' # Week 3 of 2020
    end_week = '2020W05' # Week 5 of 2020
    data = climate_health.fetch.gee_era5(credentials, polygons, start_week, end_week, band_names)
    print(data)
```
