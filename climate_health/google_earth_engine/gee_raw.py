import ee
from pydantic import BaseModel

from climate_health.api_types import FeatureCollectionModel
from climate_health.time_period import TimePeriod, PeriodRange

class GEECredentials(BaseModel):
    account: str
    private_key: str


class ERA5Entry(BaseModel):
    location: str
    period: str
    band: str
    value: float


def fetch_single_period(polygons: FeatureCollectionModel, start_dt, end_dt, band_names, reducer='mean') -> ERA5Entry:
    collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select(band_names)
    scale = collection.first().select(0).projection().nominalScale()
    features = ee.FeatureCollection(polygons.model_dump())
    dataset = collection.filterDate(start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
    # Reduce the dataset to the mean for the given region and date range
    mean_data = getattr(dataset, reducer)().reduceRegions(
            reducer=ee.Reducer.mean(),
            collection=features,
            scale=scale,  # Adjust scale based on resolution needs
        ).getInfo()
    return {feature['id']: feature['properties'] for feature in mean_data['features']}


def fetch_era5_data(credentials: GEECredentials,
                    polygons: FeatureCollectionModel,
                    start_period: str,
                    end_period: str,
                    band_names=list[str],
                    reducer: str='mean') -> list[ERA5Entry]:

    '''
    Fetch ERA5 data for the given polygons and time periods.

    Example:
        credentials = GEECredentials(account='account', private_key='private_key')
        polygons = FeatureCollectionModel(type='FeatureCollection', features=[...])
        start_period = '202001' # January 2020
        end_period = '202011' # December 2020
        band_names = ['temperature_2m', 'total_precipitation_sum']
        data = fetch_era5_data(credentials, polygons, start_period, end_period, band_names)
        assert len(data) == len(polygons.features) * len(band_names) * 11

        start_week = '2020W03' # Week 3 of 2020
        end_week = '2020W05' # Week 5 of 2020
        data = fetch_era5_data(credentials, polygons, start_week, end_week, band_names)
        assert len(data) == len(polygons.features) * len(band_names) * 3
    '''
    ee.Initialize(ee.ServiceAccountCredentials(credentials.account, key_data=credentials.private_key))
    start = TimePeriod.from_id(start_period)
    end = TimePeriod.from_id(end_period)
    period_range = PeriodRange.from_time_periods(start, end)
    data = []
    for period in period_range:
        start_day = period.start_timestamp.date
        end_day = period.last_day.date
        res = fetch_single_period(polygons, start_day, end_day, band_names, reducer=reducer)
        data.extend(ERA5Entry(location=loc, period=period.id, band=band, value=value)
                    for loc, properties in res.items() for band, value in properties.items())
    return data