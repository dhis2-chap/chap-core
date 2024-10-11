import json

import pandas as pd

from chap_core.geometry import get_area_polygons

country_name = 'BRAZIL'
df = pd.read_csv('brazil_weekly_data.csv')
admin_1_names = list(df['adm_1_name'].unique())
polygons = get_area_polygons(country_name, admin_1_names)
json.dump(polygons.model_dump(), open('brazil_polygons.json', 'w'))
