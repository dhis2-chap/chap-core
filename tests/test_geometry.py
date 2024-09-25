from chap_core.geometry import get_area_polygons


def test_get_area_polygons():
    feature_collection = get_area_polygons("norway", ["Oslo", "Akershus"])
    assert len(feature_collection.features) == 2
    assert feature_collection.features[0].id == "Oslo"
