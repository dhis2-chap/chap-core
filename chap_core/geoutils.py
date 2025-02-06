'''
Utility functions for working with geometries.
'''
import io

from .geometry import Polygons
from .api_types import FeatureModel

def feature_bbox(feature : FeatureModel):
    '''
    Calculates the bounding box for a FeatureModel object.

    Parameters
    ----------
    feature : FeatureModel
        A `FeatureModel` object representing a feature with a geometry.

    Returns
    -------
    tuple
        A 4-tuple in the form of (xmin,ymin,xmax,ymax)
    '''
    geom = feature.geometry

    geotype = geom.type
    coords = geom.coordinates

    if geotype == "Point":
        x,y = coords
        bbox = [x,y,x,y]
    elif geotype in ("MultiPoint","LineString"):
        xs, ys = zip(*coords)
        bbox = [min(xs),min(ys),max(xs),max(ys)]
    elif geotype == "MultiLineString":
        xs = [x for line in coords for x,y in line]
        ys = [y for line in coords for x,y in line]
        bbox = [min(xs),min(ys),max(xs),max(ys)]
    elif geotype == "Polygon":
        exterior = coords[0]
        xs, ys = zip(*exterior)
        bbox = [min(xs),min(ys),max(xs),max(ys)]
    elif geotype == "MultiPolygon":
        xs = [x for poly in coords for x,y in poly[0]]
        ys = [y for poly in coords for x,y in poly[0]]
        bbox = [min(xs),min(ys),max(xs),max(ys)]
    return bbox

def render(polygons : Polygons):
    '''
    Simple utility to render a `Polygons` object on a map for inspecting and debugging purposes.

    Parameters
    ----------
    polygons : Polygons
        A `Polygons` object representing the set of polygons to be rendered.

    Returns
    -------
    PIL.Image.Image
        The rendered map image. 
    '''
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from PIL import Image

    df = gpd.GeoDataFrame.from_features(polygons.__geo_interface__)
    fig, ax = plt.subplots(dpi=300)
    df.plot(ax=ax)

    # Save to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight")
    plt.close(fig)  # Close figure

    # Load image from memory buffer
    buf.seek(0)
    img = Image.open(buf)
    return img

def simplify_topology(polygons : Polygons, threshold=None):
    '''
    Simplifies a `Polygons` object while preserving topology between adjacent polygons.

    Parameters
    ----------
    polygons : Polygons
        A `Polygons` object representing the set of polygons to be simplified.
    threshold : float, optional
        Coordinate distance threshold used to simplify/round coordinates. If None, the distance 
        threshold will be automatically calculated relative to the bounding box of all polygons, 
        specifically one-thousandth of the longest of the bounding box width or height. 
        The threshold distance is specified in coordinate units. For latitude-longitude coordinates, 
        the threshold should be specified in decimal degrees, where 0.01 decimal degrees is roughly 
        1 km at the equator but increases towards the poles. 
        For more accurate thresholds, the Polygons object should be created using projected coordinates 
        instead of latitude-longitude.

    Returns
    -------
    Polygons
        A simplified `Polygons` object with preserved topology.
    '''
    import topojson as tp

    # auto calc threshold if not given
    if not threshold:
        # calc as 1 thousandth of longest width or height of all polygons
        frac = 0.001
        xmin,ymin,xmax,ymax = polygons.bbox
        w,h = xmax-xmin, ymax-ymin
        longest = max(w,h)
        threshold = longest * frac

    # generate topology and simplify
    # This is where the topology is created and simplified. Right now only sets the toposimplify parameter, 
    # which sets the distance threshold used for simplifying. Other parameters that are also relevant and 
    # might need to be specified in the future are prequantize, presimplify, and topoquantize.
    # See https://mattijn.github.io/topojson/example/settings-tuning.html#prevent_oversimplify.
    kwargs = {
        'toposimplify': threshold,
        'prevent_oversimplify': True,
        #'simplify_with': 'simplification',
    }
    topo = tp.Topology(polygons.__geo_interface__, **kwargs)

    # convert back to geojson
    geoj = topo.__geo_interface__

    # return new Polygons object
    return Polygons.from_geojson(geoj)
