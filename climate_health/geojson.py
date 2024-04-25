import geopandas as gpd


def geojson_to_shape(geojson_filename: str, shape_filename: str):
    gdf = gpd.read_file(geojson_filename)
    gdf.to_file(shape_filename)
