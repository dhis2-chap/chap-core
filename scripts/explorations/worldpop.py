import json
import time
from pydantic_geojson import  MultiPolygonModel, PolygonModel
import requests

from chap_core.api_types import FeatureCollectionModel, FeatureModel


geojson_path  = '/home/knut/Sources/climate_health/vietnam_polygons_simple.geojson'
geojson = FeatureCollectionModel.model_validate_json(open(geojson_path).read())

def convert_geometry(feature: FeatureModel):
    if isinstance(feature.geometry, MultiPolygonModel):
        coord_len, coord = max([(len(polygon), polygon) for polygon in feature.geometry.coordinates])
        polygon = PolygonModel(coordinates=coord)
        return FeatureModel(geometry=polygon, id=feature.id, properties=feature.properties)
    return feature


geojson.features = geojson.features
geojson.features = [convert_geometry(feature) for feature in geojson.features]

def get_population(feature):
    feature_collection = FeatureCollectionModel(
        type="FeatureCollection",
        features=[feature]
    )
    your_geojson = feature_collection.model_dump_json()
    api_path = f'https://api.worldpop.org/v1/services/stats?dataset=wpgppop&year=2020&geojson={your_geojson}'
    response = requests.get(api_path)
    if response.status_code != 200:
        raise ValueError(response)
    taskid = response.json()['taskid']
    result_url = f'https://api.worldpop.org/v1/tasks/{taskid}'
    for i in range(10):
        result = requests.get(result_url).json()
        if result['status'] == 'finished':
            if result['error']:
                raise ValueError(result)
            return result['data']['total_population']
        time.sleep(10)
    
results = {}
for feature in geojson.features:
    results[feature.id] = get_population(feature)
    print(feature.id, results[feature.id])
json.dump(results, open('vietnam_population.json', 'w'))