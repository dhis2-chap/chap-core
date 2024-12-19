import os

import ee
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

from chap_core.api_types import FeatureCollectionModel
from chap_core.time_period import TimePeriod, PeriodRange
import logging

logger = logging.getLogger(__name__)


class GEECredentials(BaseModel):
    account: str
    private_key: str


class ERA5Entry(BaseModel):
    location: str
    period: str
    band: str
    value: float


def load_credentials() -> GEECredentials:
    """
    Load Google Earth Engine credentials from the environment variables.

    Returns
    -------
    GEECredentials
        The Google Earth Engine credentials.
    """

    load_dotenv(find_dotenv())
    account = os.environ.get("GOOGLE_SERVICE_ACCOUNT_EMAIL")
    private_key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY")
    if private_key is None:
        raise ValueError("Google Earth Engine private key not found in environment variables")
    return GEECredentials(account=account, private_key=private_key)


def fetch_single_period(polygons: FeatureCollectionModel, start_dt, end_dt, band_names, reducer="mean") -> ERA5Entry:
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(band_names)
    scale = collection.first().select(0).projection().nominalScale()
    json_features = polygons.model_dump()
    features = ee.FeatureCollection(json_features)
    dataset = collection.filterDate(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    # Reduce the dataset to the mean for the given region and date range

    mean_data = (
        getattr(dataset, reducer)()
        .reduceRegions(
            reducer=ee.Reducer.mean(),
            collection=features,
            scale=scale,  # Adjust scale based on resolution needs
        )
        .getInfo()
    )
    return {feature["id"]: feature["properties"] for feature in mean_data["features"]}


def fetch_era5_data_generator(
        credentials: GEECredentials | dict[str, str],
        polygons: FeatureCollectionModel | str,
        start_period: str,
        end_period: str,
        band_names=list[str],
        reducer: str = "mean",
) -> list[ERA5Entry]:
    """
    Fetch ERA5 data for the given polygons, time periods, and band names.

    Parameters
    ----------
    credentials : GEECredentials
        The Google Earth Engine credentials to use for fetching the data.
    polygons : FeatureCollectionModel
        The polygons to fetch the data for.
    start_period : str
        The start period to fetch the data for.
    end_period : str
        The end period (last period) to fetch the data for.
    band_names : list[str]
        The band names to fetch the data for.

    Returns
    -------
    list[ERA5Entry]
        The fetched ERA5 data in long format

    Examples
    --------

        >>> import chap_core.fetch
        >>> credentials = GEECredentials(account="demoaccount@demo.gserviceaccount.com", private_key="private_key")
        >>> polygons = FeatureCollectionModel(type="FeatureCollection", features=[...])
        >>> start_period = "202001"  # January 2020
        >>> end_period = "202011"  # December 2020
        >>> band_names = ["temperature_2m", "total_precipitation_sum"]
        >>> data = chap.fetch.gee_era5(credentials, polygons, start_period, end_period, band_names)
        >>> assert len(data) == len(polygons.features) * len(band_names) * 11
        >>> start_week = "2020W03"  # Week 3 of 2020
        >>> end_week = "2020W05"  # Week 5 of 2020
        >>> data = fetch_era5_data(credentials, polygons, start_week, end_week, band_names)
        >>> assert len(data) == len(polygons.features) * len(band_names) * 3

    """
    if isinstance(credentials, dict):
        credentials = GEECredentials(**credentials)
    if isinstance(polygons, str):
        polygons = FeatureCollectionModel.model_validate_json(polygons)
    ee.Initialize(ee.ServiceAccountCredentials(credentials.account, key_data=credentials.private_key))
    start = TimePeriod.from_id(start_period)
    end = TimePeriod.from_id(end_period)
    period_range = PeriodRange.from_time_periods(start, end)
    # data = []
    n_periods = len(period_range)
    logger.info(f"Fetching gee data for {n_periods} periods")
    n_simultaneous_requests = 50
    polygon_subsets = [
        FeatureCollectionModel(type=polygons.type, features=polygons.features[i:i + n_simultaneous_requests])
        for i in range(0, len(polygons.features), n_simultaneous_requests)]

    for i, period in enumerate(period_range):
        logger.info(f"Fetching period {period}: {i + 1}/{n_periods}")
        start_day = period.start_timestamp.date
        end_day = period.last_day.date
        for j, subset_polygon in enumerate(polygon_subsets):
            logger.info(f"Fetching subset {j + 1}/{len(polygon_subsets)}")
            res = fetch_single_period(subset_polygon, start_day, end_day, band_names, reducer=reducer)

            items = (ERA5Entry(location=loc, period=period.id, band=band, value=value) for loc, properties in
                     res.items() for band, value in properties.items())
            yield from items

    # return data


def fetch_era5_data(
        credentials: GEECredentials | dict[str, str],
        polygons: FeatureCollectionModel | str,
        start_period: str,
        end_period: str,
        band_names=list[str],
        reducer: str = "mean",
) -> list[ERA5Entry]:
    return list(fetch_era5_data_generator(credentials, polygons, start_period, end_period, band_names, reducer=reducer))
