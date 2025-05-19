import json
import sys

import pandas as pd

from chap_core.geometry import Polygons, get_area_polygons

country_name = 'GHANA'
#df = pd.read_csv('brazil_weekly_data.csv')
#admin_1_names = list(df['adm_1_name'].unique())
polygons = get_area_polygons(country_name, admin_level=1)
polygons = Polygons(polygons)
polygons.to_file('ghana.geojson')
#json.dump(polygons.model_dump(), open('brazil_polygons.json', 'w'))

