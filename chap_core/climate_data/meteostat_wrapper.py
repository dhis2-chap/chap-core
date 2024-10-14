from datetime import datetime

import pandas as pd
from meteostat import Point, Daily, Monthly

from chap_core.geo_coding.location_lookup import LocationLookup
from chap_core.services.cache_manager import get_cache


class ClimateDataMeteoStat:
    """
    Look up weather date from start, end date delta and location
    """

    def __init__(self):
        """
        Initialize the data dictionary and location
        """
        self.data: pd.DataFrame = pd.DataFrame()
        self._delta: str = None

    def get_climate(self, location: str, start_date_string: str, end_date_string: str) -> pd.DataFrame:
        """
        Get the climate data for the given location, start date, end date and time period
        """

        # Format date
        start_date = self._format_start_date(start_date_string)
        end_date = self._format_end_date(end_date_string)

        cache_data = self._get_cache_climate(location, start_date_string, end_date_string)
        if cache_data is not None:
            return cache_data

        # Fetch location
        location_point = self._fetch_location(location)

        climate_dataframe = self._fetch_climate_data(location_point, start_date, end_date)
        climate_dataframe = self._aggregate_climate_data(climate_dataframe)
        climate_dataframe = self._format_index(climate_dataframe)

        self._add_cache_climate(location, climate_dataframe)

        return climate_dataframe

    def _format_index(self, climate_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Format the index of the dataframe
        """
        if self._delta == "day":
            climate_dataframe = climate_dataframe.rename_axis("time_period").reset_index()
            climate_dataframe["time_period"] = (
                climate_dataframe["time_period"].dt.strftime("%Y-%m-%d").str.replace("-0", "-")
            )

        elif self._delta == "week":
            climate_dataframe["time_period"] = climate_dataframe.index.to_series().apply(lambda x: x.strftime("%G-W%V"))
            climate_dataframe["time_period"] = climate_dataframe["time_period"].str.replace("-W0", "-W")

            # Set 'Year-Week' as new index
            climate_dataframe.set_index("time_period", inplace=True)
            climate_dataframe.reset_index(inplace=True)

        elif self._delta == "month":
            climate_dataframe = climate_dataframe.rename_axis("time_period").reset_index()
            climate_dataframe["time_period"] = (
                climate_dataframe["time_period"].dt.strftime("%Y-%m").str.replace("-0", "-")
            )

        elif self._delta == "year":
            climate_dataframe = climate_dataframe.rename_axis("time_period").reset_index()
            climate_dataframe["time_period"] = climate_dataframe["time_period"].astype(str)

        return climate_dataframe

    def _aggregate_climate_data(self, climate_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the climate data based on the time period for weekly and yearly data
        """
        if self._delta == "week":
            climate_dataframe = climate_dataframe.resample("W").agg(
                {
                    "rainfall": "sum",  # Sum for precipitation
                    "mean_temperature": "mean",  # Mean for average temperature
                    "max_temperature": "max",  # Max for maximum temperature
                }
            )

        elif self._delta == "year":
            climate_dataframe = climate_dataframe.groupby(climate_dataframe.index.year).agg(
                {
                    "rainfall": "sum",  # Sum the precipitation values
                    "mean_temperature": "mean",  # Average the average temperature values
                    "max_temperature": "max",  # Maximum of the maximum temperature values
                }
            )

        climate_dataframe = climate_dataframe.round(1)

        return climate_dataframe

    def _fetch_climate_data(self, location: Point, start_date: datetime, end_date: datetime):
        """
        Fetch the climate data for the given location, start date and end date
        """
        if self._delta == "day" or self._delta == "week":
            climate_data = Daily(location, start_date, end_date).fetch()
        elif self._delta == "month" or self._delta == "year":
            climate_data = Monthly(location, start_date, end_date).fetch()
        else:
            raise ValueError("Invalid time period")

        # self.data = climate_data[['prcp', 'tavg', 'tmax']].rename(
        #     columns={'prcp': 'rainfall', 'tavg': 'mean_temperature', 'tmax': 'max_temperature'})

        return climate_data[["prcp", "tavg", "tmax"]].rename(
            columns={
                "prcp": "rainfall",
                "tavg": "mean_temperature",
                "tmax": "max_temperature",
            }
        )

    def _fetch_location(self, location: str) -> Point:
        """
        Fetch the location from the given location string
        """
        location_lookup = LocationLookup()
        location = location_lookup[location]

        return Point(round(location.latitude, 4), round(location.longitude, 4), 0)

    def _format_start_date(self, start_date: str) -> datetime:
        """
        Format the string date to a datetime object
        """
        if "W" in start_date:
            self._delta = "week"
            return datetime.strptime(start_date + "-1", "%Y-W%W-%w")
        return self._format_standard_date(start_date)

    def _format_end_date(self, end_date: str) -> datetime:
        """
        Format the time period string to a datetime object
        """
        if "W" in end_date:
            self._delta = "week"
            return datetime.strptime(end_date + "-0", "%Y-W%W-%w")
        return self._format_standard_date(end_date)

    def _format_standard_date(self, date: str) -> datetime:
        """
        Format the time period string to a datetime object
        """
        if len(date.split("-")) == 3:
            self._delta = "day"
            date_format = "%Y-%m-%d"
        elif len(date.split("-")) == 2:
            self._delta = "month"
            date_format = "%Y-%m"
        elif len(date.split("-")) == 1:
            self._delta = "year"
            date_format = "%Y"
        return datetime.strptime(date, date_format)

    def _generate_cache_key(self, location):
        return f"{location}_{self._delta}"

    def _add_cache_climate(self, location: str, data: pd.DataFrame) -> None:
        """
        Add the climate data to the cache. If key already exists only add missing rows
        """
        cache = get_cache()
        cache_key = self._generate_cache_key(location)
        cache_data = cache.get(cache_key)
        # If there are already values for the key merge the new data with the old data
        if cache_data is not None:
            data = pd.concat([cache_data, data]).drop_duplicates(subset="time_period")
        cache.set(cache_key, data)

    def _get_cache_climate(self, location: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """
        Get the climate data from the cache only if all date from start to end are present
        """
        cache = get_cache()
        cache_key = self._generate_cache_key(location)
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            date_list = self._make_date_range(start_date, end_date)
            time_period_set = set(cached_data["time_period"])
            if all(date in time_period_set for date in date_list):
                return cached_data[cached_data["time_period"].isin(date_list)]
        return None

    def _make_date_range(self, start_date: str, end_date: str) -> list:
        """
        Get list of dates between start and end date
        """
        if self._delta == "day":
            date_range = pd.date_range(start=start_date, end=end_date)
            return [date.strftime("%Y-%m-%d").replace("-0", "-") for date in date_range]
        elif self._delta == "week":
            start_week = pd.to_datetime(start_date + "-1", format="%Y-W%W-%w")
            end_week = pd.to_datetime(end_date + "-1", format="%Y-W%W-%w")
            date_range = pd.date_range(start=start_week, end=end_week, freq="W-MON")
            return [f"{date.isocalendar()[0]}-W{date.isocalendar()[1]}" for date in date_range]
        elif self._delta == "month":
            date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
            return [date.strftime("%Y-%m").replace("-0", "-") for date in date_range]
        elif self._delta == "year":
            return [str(year) for year in range(int(start_date), int(end_date) + 1)]
