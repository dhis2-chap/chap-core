from matplotlib import pyplot as plt

from chap_core.climate_data.harmonization import harmonize_health_dataset
from chap_core.data.open_dengue import OpenDengueDataSet
from chap_core.datatypes import HealthData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

def filter_country(country_name, spatial_resolution):
    dataset = OpenDengueDataSet().as_dataset(country_name, spatial_resolution=spatial_resolution,
                                             temporal_resolution='Week')
    dataset = DataSet.from_pandas(dataset, HealthData, fill_missing=True)
    dataset.to_csv(f'{country_name}_weekly_cases_{spatial_resolution}.csv')
    dataset.plot()
    plt.show()
    return dataset

if __name__ == '__main__':
    filter_country('BOLIVIA', 'Admin1')
