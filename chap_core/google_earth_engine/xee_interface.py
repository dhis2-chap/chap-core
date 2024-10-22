import ee
import xarray

from chap_core.google_earth_engine.gee_raw import load_credentials


class XeeInterface:
    def __init__(self):
        self.credentials = load_credentials()
        ee.Initialize(ee.ServiceAccountCredentials(self.credentials.account, key_data=self.credentials.private_key))

    def get_data(self, start_date: str, end_date: str, polygon: dict) -> xarray.Dataset:
        collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(start_date, end_date).select(
            'temperature_2m', 'total_precipitation_sum')
        return xarray.open_dataset(
            collection,
            engine='ee',
            projection=collection.first().select(0).projection(),
            geometry=ee.Geometry(polygon)
        )
