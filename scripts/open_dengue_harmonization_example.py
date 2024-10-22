from chap_core.google_earth_engine.gee_raw import load_credentials
import pandas as pd
import pooch  # type: ignore
from dateutil.parser import parse
import pickle
from chap_core.data.open_dengue import OpenDengueDataSet
import json
from pydantic.json import pydantic_encoder
import logging

logging.basicConfig(level=logging.INFO)
import ee
import xarray

from chap_core.fetch import get_area_polygons, gee_era5
from chap_core.time_period import Week, PeriodRange

# Fetch the open dengue data
country_name = 'BRAZIL'
if True:
    data_path = 'https://github.com/OpenDengue/master-repo/raw/main/data/releases/V1.2.2/Temporal_extract_V1_2_2.zip'
    zip_filename = pooch.retrieve(data_path, None)

    # Read the data

    df = pd.read_csv(zip_filename, compression='zip')

    df = df[df['adm_0_name'] == country_name]
    df = df[df['T_res'] == 'Week']
    df = df[df['adm_1_name'].notna()]
    start_dates = [parse(date) for date in df['calendar_start_date']]
    # Filter out the monday weeks since there are more sundays
    df = df[[date.weekday() == 6 for date in start_dates]]
    #df.to_csv('brazil_weekly_data.csv', index=False)
else:
    df = pd.read_csv('brazil_weekly_data.csv')
if False:
    admin_1_names = list(df['adm_1_name'].unique())
    polygons = get_area_polygons(country_name, admin_1_names)

    min_date = df['calendar_start_date'].min()
    max_date = df['calendar_end_date'].max()

    min_start_date = parse(df['calendar_start_date'].min())
    max_start_date = parse(df['calendar_start_date'].max())
    first_week = Week(min_start_date)
    last_week = Week(max_start_date)
    # period_range = PeriodRange.from_time_periods(first_week, last_week)


    credentials = load_credentials()
    data = gee_era5(credentials, polygons, first_week.id, last_week.id, ['temperature_2m', 'total_precipitation_sum'])
    pickle.dump(data, open('brazil_era5_data.pkl', 'wb'))
    json.dumps(polygons, open('brazil_polygons.json', 'w'),
               default=pydantic_encoder)
if False:
    max_date = df.calendar_end_date.max()
    service_account = 'dhis2-demo@dhis2-gis.iam.gserviceaccount.com'
    # credentials = ee.ServiceAccountCredentials(service_account, '/Users/mastermaps/DHIS2/dhis-google-auth.json')
    # credentials = load_credentials()
    ee.Initialize(ee.ServiceAccountCredentials(credentials.account, key_data=credentials.private_key))
    collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(min_date, max_date).select(
        'temperature_2m', 'total_precipitation_sum')
    projection = collection.first().select(0).projection()
    credentials = load_credentials()
    dataset = None
    for feature in polygons.model_dump()['features']:
        geometry = feature['geometry']
        dataset = xarray.open_dataset(
            collection,
            engine='ee',
            projection=projection,
            geometry=ee.Geometry(geometry)
        )
        break
if __name__ == '__main__':
    base_df = OpenDengueDataSet().as_dataset('BRAZIL', spatial_resolution='Admin2', temporal_resolution='Week')
    polygons = get_area_polygons('brazil', base_df['location'].unique(), 2)
    credentials = load_credentials()
    first_week,last_week = min(base_df['time_period']), max(base_df['time_period'])
    data = gee_era5(credentials, polygons, first_week.id, last_week.id, ['temperature_2m', 'total_precipitation_sum'])
    pickle.dump(data, open('brazil_era5_data_adm2.pkl', 'wb'))
if False:
    time_periods = [Week(parse(date)) for date in base_df['calendar_start_date']]
    base_df.time_period = time_periods

    brazil_data = pickle.load(open('brazil_era5_data.pkl', 'rb'))
    df = pd.DataFrame([de.model_dump() for de in brazil_data])
    # Make one column for each value of 'band'
    df = df.pivot(index=('period', 'location'), columns='band', values='value')
    # Reset the index to make the columns 'period' and 'location' into regular columns
    df = df.reset_index()
    # Rename the columns
    df = df.rename(
        columns={'period': 'time_period', 'temperature_2m': 'mean_temperature', 'total_precipitation_sum': 'rainfall'})

    # df2 = pd.read_csv('brazil_weekly_data.csv')

