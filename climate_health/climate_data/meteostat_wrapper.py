from datetime import datetime

import pandas as pd
from meteostat import Point, Daily, Monthly

from climate_health.geo_coding.location_lookup import LocationLookup



#TODO: Simplify the class
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


    def get_climate(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get the climate data for the given location, start date, end date and time period
        """

        # Format date
        start_date = self._format_start_date(start_date)
        end_date = self._format_end_date(end_date)


        # Fetch location
        location = self._fetch_location(location)


        # Fetch climate data
        self.data = self._fetch_climate_data(location, start_date, end_date)

        self._aggregate_climate_data()

        self._format_index()





    def _format_index(self):
        """
        Format the index of the dataframe
        """
        if self._delta == 'day':
            self.data = self.data.rename_axis('time_period').reset_index()
            self.data['time_period'] = self.data['time_period'].dt.strftime('%Y-%m-%d').str.replace('-0', '-')

        elif self._delta == 'week':
            self.data['time_period'] = self.data.index.to_series().apply(lambda x: x.strftime('%G-W%V'))
            self.data['time_period'] = self.data['time_period'].str.replace('-W0', '-W')

            # Set 'Year-Week' as new index
            self.data.set_index('time_period', inplace=True)
            self.data.reset_index(inplace=True)

        elif self._delta == 'month':
            self.data = self.data.rename_axis('time_period').reset_index()
            self.data['time_period'] = self.data['time_period'].dt.strftime('%Y-%m').str.replace('-0', '-')

        elif self._delta == 'year':
            self.data = self.data.rename_axis('time_period').reset_index()
            self.data['time_period'] = self.data['time_period'].astype(str)




    def _aggregate_climate_data(self):
        """
        Aggregate the climate data based on the time period for weekly and yearly data
        """
        if self._delta == 'week':
            self.data = self.data.resample('W').agg({
                'rainfall': 'sum',  # Sum for precipitation
                'mean_temperature': 'mean',  # Mean for average temperature
                'max_temperature': 'max'  # Max for maximum temperature
            })

        elif self._delta == 'year':
            self.data = self.data.groupby(self.data.index.year).agg({
                'rainfall': 'sum',  # Sum the precipitation values
                'mean_temperature': 'mean',  # Average the average temperature values
                'max_temperature': 'max'  # Maximum of the maximum temperature values
            })

        self.data = self.data.round(1)



    def _fetch_climate_data(self, location: Point, start_date: datetime, end_date: datetime):
        """
        Fetch the climate data for the given location, start date and end date
        """
        if self._delta == 'day' or self._delta == 'week':
            climate_data = Daily(location, start_date, end_date).fetch()
        elif self._delta == 'month' or self._delta == 'year':
            climate_data = Monthly(location, start_date, end_date).fetch()
        else:
            raise ValueError('Invalid time period')

        # self.data = climate_data[['prcp', 'tavg', 'tmax']].rename(
        #     columns={'prcp': 'rainfall', 'tavg': 'mean_temperature', 'tmax': 'max_temperature'})

        return climate_data[['prcp', 'tavg', 'tmax']].rename(
            columns={'prcp': 'rainfall', 'tavg': 'mean_temperature', 'tmax': 'max_temperature'})


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
        if 'W' in start_date:
            self._delta = 'week'
            return datetime.strptime(start_date + '-1', '%Y-W%W-%w')
        return self._format_standard_date(start_date)


    def _format_end_date(self, end_date: str) -> datetime:
        """
        Format the time period string to a datetime object
        """
        if 'W' in end_date:
            self._delta = 'week'
            return datetime.strptime(end_date + '-0', '%Y-W%W-%w')
        return self._format_standard_date(end_date)


    def _format_standard_date(self, date: str) -> datetime:
        """
        Format the time period string to a datetime object
        """
        if len(date.split('-')) == 3:
            self._delta = 'day'
            date_format = '%Y-%m-%d'
        elif len(date.split('-')) == 2:
            self._delta = 'month'
            date_format = '%Y-%m'
        elif len(date.split('-')) == 1:
            self._delta = 'year'
            date_format = '%Y'
        return datetime.strptime(date, date_format)


    def __str__(self):
        return str(self.data)



    def equals(self, other):
        """
        Define equality as the .data attribute being equal to the provided DataFrame with a rounding error of 0.01
        """
        try:
            pd.testing.assert_frame_equal(self.data, other, check_exact=False, atol=0.01)
            return True
        except AssertionError:
            return False
