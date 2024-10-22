import json

import geopandas
import matplotlib.pyplot as plt
import xarray
#import zarr
import pytest

from chap_core.api_types import FeatureCollectionModel
from chap_core.google_earth_engine.xee_interface import XeeInterface

@pytest.fixture
def brazil_polygons(data_path):
    return json.load(open(data_path/'brazil_polygons.json'))

@pytest.fixture
def rio_polygon(brazil_polygons):
    polygons = FeatureCollectionModel.model_validate(brazil_polygons)
    rio = next(f for f in polygons.features if f.id == 'RiodeJaneiro')
    return rio

@pytest.mark.skip
def test_get_data(rio_polygon):
    xee = XeeInterface()
    data = xee.get_data('2020-01-01', '2020-01-02', rio_polygon.geometry.model_dump())
    data.to_zarr('rio_data.zarr')

@pytest.mark.skip
def test_plot_data(data_path):
    df = geopandas.read_file(data_path/'brazil_polygons.json')
    df.plot()
    plt.show()
    data = xarray.open_zarr('rio_data.zarr')
    data['temperature_2m'].plot()
    plt.show()
    # plt.show()
    # plot polygon
    #features = geopandas.GeoDataFrame.from_features([rio_polygon.model_dump()])
    #features.plot()
