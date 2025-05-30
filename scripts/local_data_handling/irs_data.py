import numpy as np

from chap_core.datatypes import create_tsdataclass, FullData, tsdataclass
from chap_core.geometry import Polygons
from chap_core.spatio_temporal_data.converters import observations_to_dataset
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

@tsdataclass
class IRSData(FullData):
    irs: float

filename = '/home/knut/Downloads/irs_pop_cases_temp.json'
from chap_core.rest_api_src.data_models import DatasetMakeRequest
request = DatasetMakeRequest.parse_file(filename)
feature_names = list({entry.feature_name for entry in request.provided_data})
dataclass = create_tsdataclass(feature_names)
provided_data = observations_to_dataset(dataclass, request.provided_data, fill_missing=True)
old_dataset = DataSet.from_csv('/home/knut/Data/ch_data/rwanda_full_resolution.csv', FullData)
predecessors = set(provided_data.locations())
map = Polygons(old_dataset.polygons).get_predecessors_map(predecessors)
restricted = old_dataset.restrict_time_period(slice(None, provided_data.period_range[-1]))
new_data = {}
for location, data in restricted.items():
    irs = provided_data[map[location]].irs
    irs[np.isnan(irs)] = 0
    new_data[location] = IRSData(time_period=data.time_period,
                                  disease_cases=data.disease_cases,
                                  population=data.population,
                                  rainfall=data.rainfall,
                                  mean_temperature=data.mean_temperature,
                                  irs=irs)
new_dataset = DataSet(new_data)
new_dataset.set_polygons(old_dataset.polygons)
new_dataset.to_csv('/home/knut/Data/ch_data/rwanda_full_resolution_irs.csv')