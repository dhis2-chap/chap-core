from matplotlib import pyplot as plt

from chap_core.data.open_dengue import OpenDengueDataSet
from chap_core.datatypes import HealthData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

for spatial_resolution in ['Admin2']:
    dataset = OpenDengueDataSet().as_dataset('COLOMBIA', spatial_resolution=spatial_resolution,
                                             temporal_resolution='Week')
    if __name__ == '__main__':
        set = DataSet.from_pandas(dataset, HealthData, fill_missing=True)
        set.to_csv(f'colombia_weekly_cases_{spatial_resolution}.csv')


