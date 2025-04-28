from chap_core.data import DataSet
from chap_core.datatypes import FullData
from chap_core.geometry import Polygons

data_path = '/home/knut/Data/ch_data/rwanda_full_resolution.csv'
geojson = '/home/knut/Data/ch_data/rwanda_full_resolution.geojson'
ds = DataSet.from_csv(data_path, FullData)
#ds.set_polygons(Polygons.from_file(geojson).feature_collection())
#df = ds.to_pandas()
ds.to_report('rwanda_full_resolution.pdf')
#print(df)
#feature_collection = Polygons.from_file(geojson).feature_collection()

#parent_dict = ds.get_parent_dict()
#print(parent_dict)
