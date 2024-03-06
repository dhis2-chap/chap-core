from climate_health.datatypes import Location
from climate_health.geo_coding.location_lookup import LocationLookup
import pytest


@pytest.mark.xfail
def test_location_lookup_contains():
    location_lookup = LocationLookup()
    assert 'Oslo' in location_lookup
    assert 'Paris' in location_lookup
    assert 'MadeUpLocation' not in location_lookup


@pytest.mark.xfail
def test_location_lookup_getitem():
    location_lookup = LocationLookup()
    assert location_lookup['Oslo'] == Location(59.9133301,
                                               10.7389701)  # These are values from copilot, might not be correct
    assert location_lookup['Paris'] == Location(48.8566, 2.3522)


@pytest.mark.xfail
def test_raises_key_error():
    location_lookup = LocationLookup()
    with pytest.raises(KeyError) as e:
        location_lookup['MadeUpLocation']
