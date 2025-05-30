import numpy as np

from chap_core.data import DataSet

filename = '../../historic_data.csv'

data = DataSet.from_csv(filename)
for location, location_data in data.items():
    assert not np.all(np.isnan(location_data.disease_cases)), (location)
