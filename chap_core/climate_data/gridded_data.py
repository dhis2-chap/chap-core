import ee
import xarray
from matplotlib import pyplot as plt

from ..google_earth_engine.gee_raw import load_credentials
import geopandas as gpd


# Load the GeoJSON file using GeoPandas
def get_gridded_data(polygons_filename):
    gdf = gpd.read_file(polygons_filename)
    # Get the bounding box of all polygons in the GeoJSON
    lon1, lat1, lon2, lat2 = gdf.total_bounds
    print(lon1, lat1, lon2, lat2)
    credentials = load_credentials()
    ee.Initialize(ee.ServiceAccountCredentials(credentials.account, key_data=credentials.private_key))
    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate("2024-08-01", "2024-8-03").select("temperature_2m")
    )
    # lon1 = 28.8
    # lon2 = 30.9
    # lat1 = -2.9
    # lat2 = -1.0
    country_bounds = ee.Geometry.Rectangle(*gdf.total_bounds)  # lon1, lat1, lon2, lat2)
    projection = collection.first().select(0).projection()  # EPSG:4326
    dataset = xarray.open_dataset(collection, engine="ee", projection=projection, geometry=country_bounds)
    first_image = dataset.isel(time=0)
    temp_d = first_image["temperature_2m"]
    temp_d.plot()
    temp = temp_d.values
    # plt.imshow(temp, extent=[ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()], origin='lower', cmap='viridis',
    #           norm=Normalize())
    # plt.imshow(temp, cmap='viridis')
    gdf.boundary.plot(ax=plt.gca(), edgecolor="red", linewidth=1)
    plt.show()
    return temp
