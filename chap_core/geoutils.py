'''
Utility functions for working with geometries.
'''
import io

from .geometry import Polygons
from .api_types import FeatureModel

# def count_coordinates(feature : FeatureModel):
#     # TODO: not sure if needed... 
#     geom = feature.geometry

#     geotype = geom.type
#     coords = geom.coordinates

#     if geotype == "Point":
#         xs = [coords[0]]
#     elif geotype in ("MultiPoint","LineString"):
#         xs = (x for x,y in coords)
#     elif geotype == "MultiLineString":
#         xs = (x for line in coords for x,y in line)
#     elif geotype == "Polygon":
#         xs = (x for ring in coords for x,y in ring)
#     elif geotype == "MultiPolygon":
#         xs = (x for poly in coords for ring in poly for x,y in ring)
    
#     return len(list(xs))

def feature_bbox(feature : FeatureModel):
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

# def show(img):
#     # TODO: not yet working in WSL
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Convert image to NumPy array and display using Matplotlib (works in WSL)
#     img_array = np.array(img)
#     plt.imshow(img_array)
#     plt.axis("off")  # Hide axes
#     plt.show()

def toposimplify(polygons : Polygons, threshold=None):
    '''Simplifies a Polygons object while preserving topology between adjacent polygons.
    
    Args:
    - polygons: Polygons object
    - threshold: coordinate distance threshold used to simplify/round coordinates. If None, distance threshold
        will be auto calculated relative to the bbox of all polygons, specifically one-thousandth of longest of bbox width or height. 
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
    kwargs = {'toposimplify':threshold}
    topo = tp.Topology(polygons.__geo_interface__, **kwargs)

    # convert back to geojson
    geoj = topo.__geo_interface__

    # return new Polygons object
    return Polygons.from_geojson(geoj)
