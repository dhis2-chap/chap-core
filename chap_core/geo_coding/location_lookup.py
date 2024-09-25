from chap_core.datatypes import Location
from geopy.geocoders import Nominatim
from geopy.geocoders import ArcGIS

from chap_core.services.cache_manager import get_cache


class LocationLookup:
    def __init__(self, geolocator: str = "Nominatim"):
        """
        Initializes the LocationLookup object.
        """
        self.dict_location: dict = {}
        if geolocator == "ArcGIS":
            self.geolocator = ArcGIS()
        elif geolocator == "Nominatim":
            self.geolocator = Nominatim(user_agent="chap_core")

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

        if self._get_cache_location(location_name):
            return True

        if self._fetch_location(location_name):
            return True

        return False

    def __getitem__(self, location_name: str) -> Location:
        """
        Returns the Location object for the given location_name.
        """
        if location_name in self.dict_location:
            return Location(
                self.dict_location[location_name].latitude,
                self.dict_location[location_name].longitude,
            )

        if self._get_cache_location(location_name):
            return Location(
                self.dict_location[location_name].latitude,
                self.dict_location[location_name].longitude,
            )

        if self._fetch_location(location_name):
            return Location(
                self.dict_location[location_name].latitude,
                self.dict_location[location_name].longitude,
            )

        raise KeyError(location_name)

    def __str__(self) -> str:
        """
        Returns a string representation of the LocationLookup object.
        """
        return f"{self.dict_location}"

    def _generate_cache_key(self, geolocator, location_name: str) -> str:
        """
        Return a key form the cache from the location name and the geolocator used.
        """
        return f"{geolocator.domain}_{location_name}"

    def _add_cache_location(self, location_name: str, location: Location) -> None:
        """
        Add location to the cache.
        """
        cache = get_cache()
        cache_key = self._generate_cache_key(self.geolocator, location_name)
        cache[cache_key] = location

    def _get_cache_location(self, location_name: str) -> bool:
        """
        If location data was previously cached, add it to the location dictionary and resturn true.
        Else return false.
        """
        cache = get_cache()
        cache_key = self._generate_cache_key(self.geolocator, location_name)
        cached_data = cache.get(cache_key)

        if cached_data is not None:
            self.dict_location[location_name] = cached_data
            return cached_data
        else:
            return False

    def _fetch_location(self, location_name: str) -> bool:
        location = self.geolocator.geocode(location_name)
        if location is not None:
            self.dict_location[location_name] = location
            self._add_cache_location(location_name, location)
            return True
        else:
            return False

    def try_connection(self):
        self.geolocator.geocode("Oslo")
