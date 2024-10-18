import sys
from multiprocessing.managers import Value

import chap_core.fetch
from chap_core.google_earth_engine.gee_raw import load_credentials
try:
    credentials = load_credentials()#dict(account='demoaccount@demo.gserviceaccount.com', private_key='private_key')

    # polygons = open("../example_data/Organisation units.geojson").read()
    polygons = chap_core.fetch.get_area_polygons('norway', ['Oslo', 'Akershus'])
    start_period = '202001'  # January 2020
    end_period = '202011'  # December 2020
    band_names = ['temperature_2m', 'total_precipitation_sum']
    data = chap_core.fetch.gee_era5(credentials, polygons,
                                    start_period, end_period, band_names)
    print(data)
    start_week = '2020W03'  # Week 3 of 2020
    end_week = '2020W05'  # Week 5 of 2020
    data = chap_core.fetch.gee_era5(credentials, polygons, start_week, end_week, band_names)
    print(data)

except ValueError:
    print('Error, did not find credentials?')
