import json
import sys

import pandas as pd

from chap_core.file_io.example_data_set import datasets, DataSetType
from chap_core.geometry import Polygons, get_area_polygons

dataset_name = "ISIMIP_dengue_harmonized"
dataset = datasets[dataset_name]
dataset = dataset.load()
dataset = dataset["brazil"]
dataset.to_csv("brazil.csv")
print(set(dataset.keys()))
admin_1_names = list(dataset.keys())
#admin_1_names = list(dataset["adm_1_name"].unique())
#admin_1_names = list(dataset["adm_1_name"].unique())


country_name = 'BRAZIL'
#df = pd.read_csv('brazil_weekly_data.csv')
#admin_1_names = list(df['adm_1_name'].unique())
polygons = get_area_polygons(country_name, admin_1_names)
polygons = Polygons(polygons)
polygons.to_file('brazil_polygons.geojson')
#json.dump(polygons.model_dump(), open('brazil_polygons.json', 'w'))

