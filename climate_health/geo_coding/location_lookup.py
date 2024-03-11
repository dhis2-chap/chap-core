import geopy

from climate_health.datatypes import Location
from geopy.geocoders import Nominatim
from geopy.geocoders import ArcGIS



class LocationLookup:

    def __init__(self, geolocator: str = 'Nominatim'):
        """
        Initializes the LocationLookup object.
        """
        self.dict_location: dict = {}
        if geolocator == 'ArcGIS':
            self.geolocator = ArcGIS()
        elif geolocator == 'Nominatim':
            self.geolocator = Nominatim(user_agent="climate_health")



    def add_location(self, location_name: str) -> None:
        """
        Adds the location to the location lookup.
        """
        if location_name not in self.dict_location:
            location = self.geolocator.geocode(location_name)
        self.dict_location[location_name] = location


    def __contains__(self, location_name: str) -> bool:
        """
        Returns True if the location_name is we are able to geo-code the name and False otherwise.
        """
        if location_name in self.dict_location:
            return True

        # TODO: add more error handling
        # try:
        location = self.geolocator.geocode(location_name)
        if location is not None:
            self.dict_location[location_name] = location
            return True

        # except Exception as e:
        #     print(e)
        #     return False

        return False


    def __getitem__(self, location_name: str) -> Location:
        """
        Returns the Location object for the given location_name.
        """
        return Location(self.dict_location[location_name].latitude, self.dict_location[location_name].longitude)


    def __str__(self) -> str:
        """
        Returns a string representation of the LocationLookup object.
        """
        return f'{self.dict_location}'

    # def __repr__(self) -> str:
    #     """
    #     Returns a string representation of the LocationLookup object.
    #     """
    #     return f'{self.dict_location}'


