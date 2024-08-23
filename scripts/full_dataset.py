from typing import re

import geopandas
import pandas as pd
from pydantic_geojson import FeatureCollectionModel, FeatureModel
from climate_health.rest_api_src.worker_functions import initialize_gee_client
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import PeriodRange, Month

csv_file_name = '~/Data/ch_data/dengue_data_ISIMIP.csv'
polygon_filename = '/home/knut/Data/ch_data/geometry/gadm41_VNM_1.json'

class DFeatureModel(FeatureModel):
    properties: dict

class DFeatureCollectionModel(FeatureCollectionModel):
    features: list[DFeatureModel]

polygons = DFeatureCollectionModel.model_validate_json(open(polygon_filename).read())
polygon_dict = {feature.properties['VARNAME_1']: feature for feature in polygons.features}

#print(polygons)


df = pd.read_csv(csv_file_name)
#print(countries)

countries = df['country'].unique()
print(list(countries))
exit()
vietnam_entries = df[df['country'] == 'vietnam']

periods = [pd.Period(f'{year}-{month}', 'M') for year, month in zip(vietnam_entries['year'], vietnam_entries['month'])]
min_period = min(zip(vietnam_entries['year'], vietnam_entries['month']))
max_period = max(zip(vietnam_entries['year'], vietnam_entries['month']))
print(min_period, max_period)
period_range = PeriodRange.from_time_periods(Month(*min_period), Month(*max_period))
print(period_range)
exit()

print(periods)
print(vietnam_entries['admin1'].unique())


myshpfile = geopandas.read_file(polygon_filename)
utf8_geom_names = set(myshpfile['VARNAME_1'].unique())
def to_camelcase(name: str) -> str:
    return ''.join([word.capitalize() for word in name.split()])

geometry_dict = {}
geojson_dict = {}
gee_client = initialize_gee_client()
features = []
for name in vietnam_entries['admin1'].unique():
    cname = to_camelcase(name)
    print(cname, cname in polygon_dict)
    if cname in polygon_dict:
        #json = polygon_dict[cname].json()
        features.append(polygon_dict[cname])
        # json = myshpfile[myshpfile['VARNAME_1'] == cname].to_json()
        #weather = gee_client.get_historical_era5(json, periodes=PeriodRange.from_strings(['2012-01', '2012-02']))
        #geometry_dict[name] = myshpfile[myshpfile['VARNAME_1'] == cname]['geometry'].values[0]
        #geojson_dict[name] = myshpfile[myshpfile['VARNAME_1'] == cname].to_json()

print(features)
names = [feature.properties['VARNAME_1'] for feature in features]
data = gee_client.get_historical_era5(DFeatureCollectionModel(features=features).model_dump(), periodes=period_range)
data = SpatioTemporalDict({name: value for name, value in zip(names, data.values())})
data.to_csv('vietnam_era5.csv')


@dataclass
class DengueData:
    disease_cases: int
    mean_temperature: float
    rainfall: float

data = {'disease_cases': 20,
        'mean_temperature': 25,
        'rainfall': 100}



