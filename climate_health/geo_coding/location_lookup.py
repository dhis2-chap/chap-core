from climate_health.datatypes import Location


class LocationLookup:
    def __contains__(self, location_name: str) -> bool:
        '''
        Returns True if the location_name is we are able to geo-code the name and False otherwise.
        '''
        ...

    def __getitem__(self, location_name: str) -> Location:
        '''
        Returns the Location object for the given location_name.
        '''
        ...
