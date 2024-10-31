from chap_core.datatypes import FullData
from chap_core.geometry import get_area_polygons
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
dataset = DataSet.from_csv('/home/knut/Data/uganda_data.csv', FullData)
convert_name = lambda name: ' '.join(name.split()[:-1])
raw_names = dataset.locations()
place_names = list(map(convert_name, raw_names))

polygons_2 = get_area_polygons('uganda', place_names, 2)

# harmonized = harmonize_health_dataset(dataset, 'uganda', get_climate=False)
# weekly_dataset = dataset.resample('W')
# weekly_dataset.plot()
