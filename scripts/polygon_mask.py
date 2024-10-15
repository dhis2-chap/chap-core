import geopandas
import xarray
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import numpy as np


# Load your xarray dataset (assuming it's already loaded as `ds`)
# and your GeoDataFrame with the polygon (assuming it’s `gdf`)

# Extract dataset resolution based on `lat` and `lon` steps

# Initialize an empty xarray to store coverage proportions
def get_coverage_mask():
    lat_res = np.abs(ds['lat'][1] - ds['lat'][0])
    lon_res = np.abs(ds['lon'][1] - ds['lon'][0])
    coverage = xr.DataArray(
        np.zeros_like(ds['temperature_2m']),  # Assumes you want proportions for temperature data
        coords=ds['temperature_2m'].coords,
        dims=ds['temperature_2m'].dims,
    )

    # Loop over each latitude and longitude point to create pixel polygons
    for lat in ds['lat']:
        for lon in ds['lon']:
            # Create a pixel polygon centered at (lat, lon)
            pixel_polygon = box(
                lon - lon_res / 2, lat - lat_res / 2,  # lower left
                lon + lon_res / 2, lat + lat_res / 2,  # upper right
            )

            # Calculate the intersection area with the polygon in your GeoDataFrame
            intersection_area = gdf.geometry.intersection(pixel_polygon).area.sum()  # Total intersected area
            pixel_area = pixel_polygon.area  # Area of the pixel

            # Calculate coverage proportion and assign to DataArray
            coverage.loc[dict(lat=lat, lon=lon)] = intersection_area / pixel_area

    return coverage

def get_mean_values(dataset, mask):
    for var in dataset.data_vars:
        mask = ds[var].isnull()
        dataset[var] = dataset[var] * mask


ds = xarray.open_zarr('../tests/rio_data.zarr')
ds = ds.isel(time=0)
brazil_polygons = gpd.read_file('../example_data/brazil_polygons.json')
rio_polygon = brazil_polygons.loc[brazil_polygons['id'] == 'RiodeJaneiro']
gdf = rio_polygon



# Now `coverage` contains the coverage proportion of each pixel
