import ee
import xarray
import pyproj ## installed with pip install to avoid missing proj database error
import numpy as np;

from chap_core.google_earth_engine.gee_raw import load_credentials

# required: https://github.com/google/Xee
service_account = 'dhis2-demo@dhis2-gis.iam.gserviceaccount.com'
#credentials = ee.ServiceAccountCredentials(service_account, '/Users/mastermaps/DHIS2/dhis-google-auth.json')
credentials = load_credentials()
ee.Initialize(ee.ServiceAccountCredentials(credentials.account, key_data=credentials.private_key))
#ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate('2024-08-01', '2024-09-01').select('temperature_2m', 'total_precipitation_sum')
lon1 = 28.8
lon2 = 30.9
lat1 = -2.9
lat2 = -1.0
rwanda_bounds = ee.Geometry.Rectangle(lon1, lat1, lon2, lat2)
projection = collection.first().select(0).projection() # EPSG:4326
dataset = xarray.open_dataset(
    collection,
    engine='ee',
    projection=projection,
    geometry=rwanda_bounds
)
first_image = dataset.isel(time=0)
temp = first_image['temperature_2m'].values

