from chap_core.datatypes import Location
from chap_core.geo_coding.location_lookup import LocationLookup
import pytest


@pytest.fixture()
def location_lookup():
    pytest.skip()
    lookup = LocationLookup("ArcGIS")
    try:
        lookup.try_connection()
    except:
        pytest.skip()
    return lookup


@pytest.fixture()
def arcgis_lookup():
    pytest.skip()
    lookup = LocationLookup("ArcGIS")
    try:
        lookup.try_connection()
    except:
        pytest.skip()
    return lookup


# @pytest.mark.xfail
def test_location_lookup_contains_arcgis(arcgis_lookup):
    location_lookup = arcgis_lookup
    assert "Oslo" in location_lookup
    assert "Paris" in location_lookup
    assert "MadeUpLocation" not in location_lookup


# @pytest.mark.xfail
def test_location_lookup_contains_noninatime(location_lookup):
    assert "Oslo" in location_lookup
    assert "Paris" in location_lookup
    assert "MadeUpLocation" not in location_lookup


# @pytest.mark.xfail
def test_location_lookup_same_city(location_lookup):
    assert "Oslo" in location_lookup
    assert "Paris" in location_lookup
    assert "Paris, Île-de-France" in location_lookup
    assert "MadeUpLocation" not in location_lookup


def test_location_lookup_getitem_arcgis(arcgis_lookup):
    location_lookup = arcgis_lookup
    assert location_lookup["Oslo"] == Location(59.91234, 10.75)
    assert location_lookup["Paris"] == Location(48.863697576, 2.361657337)


@pytest.mark.xfail(reason="This is too accurate test, fails on CI")
def test_location_lookup_getitem_noninatime(location_lookup):
    assert location_lookup["Oslo"] == Location(59.9133301, 10.7389701)
    assert location_lookup["Paris"] == Location(48.8534951, 2.3483915)


# @pytest.mark.xfail
def test_raises_key_error(location_lookup):
    with pytest.raises(KeyError) as e:
        location_lookup["MadeUpLocation"]


def test_arcgis_geolocator(arcgis_lookup):
    location_lookup = arcgis_lookup  # LocationLookup('ArcGIS')
    assert location_lookup.geolocator.__class__.__name__ == "ArcGIS"


@pytest.fixture()
def nominatim_lookup():
    try:
        lookup = LocationLookup("Nominatim")
        lookup.try_connection()
    except:
        pytest.skip()
    return lookup


def test_nominatim_geolocator(nominatim_lookup):
    location_lookup = nominatim_lookup
    # location_lookup = LocationLookup('Nominatim')
    assert location_lookup.geolocator.__class__.__name__ == "Nominatim"


def test_print_location_lookup_arcgis(arcgis_lookup):
    location_lookup = arcgis_lookup  # LocationLookup('ArcGIS')
    location_lookup.add_location("Oslo")

    location_lookup.add_location("Paris")
    assert (
        str(location_lookup)
        == "{'Oslo': Location(Oslo, (59.91234, 10.75, 0.0)), 'Paris': Location(Paris, Île-de-France, (48.863697576, 2.361657337, 0.0))}"
    )


@pytest.mark.xfail
def test_print_location_lookup_noninatime(nominatim_lookup):
    location_lookup = nominatim_lookup
    location_lookup.add_location("Oslo")
    location_lookup.add_location("Paris")
    print(location_lookup)
    assert (
        str(location_lookup)
        == "{'Oslo': Location(Oslo, Norge, (59.9133301, 10.7389701, 0.0)), 'Paris': Location(Paris, Île-de-France, France métropolitaine, France, (48.8534951, 2.3483915, 0.0))}"
    )
