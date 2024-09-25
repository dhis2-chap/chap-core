import numpy as np
import pandas as pd

from chap_core.datatypes import ClimateData, ClimateHealthTimeSeries, FullData
from chap_core.file_io.cleaners import laos_data
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

mapping = {'rainfall': 'gsiW9SgolNd',
           'mean_temperature': 'VA05qvanuVs',
           'max_temperature': 'ZH76qVQl5Mz'}



health_filaname = '/home/knut/Downloads/dengue.csv'
df = laos_data(health_filaname)

#df.to_csv('/home/knut/Downloads/dengue_clean.csv')
health = df
climate_filename = '/home/knut/Downloads/climate_monthly_perdataelement.csv'


def get_laos_climate(climate_filename):
    climate_data = pd.read_csv(climate_filename)
    df = climate_data
    df = df.sort_values(by=['orgunit', 'year', 'month'])
    periods = [f'{year}-{month}' for year, month
               in zip(climate_data['year'],
                      climate_data['month'])]
    climate_data['periodid'] = periods
    # climate_data = climate_data.sort_values(by=['periodid'])
    d = {name: df['value.' + mapping[name]].values for name in mapping.keys()}
    new_df = pd.DataFrame(
        d | {'time_period': climate_data['periodid'], 'location': climate_data['orgunit']})
    spatio_temporal_dict = DataSet.from_pandas(
        new_df, dataclass=ClimateData)
    return spatio_temporal_dict.interpolate()

spatio_temporal_dict = get_laos_climate(climate_filename)
full_dict = {name: ClimateHealthTimeSeries.combine(health.get_location(name).data(), spatio_temporal_dict.get_location(name).data())
             for name in health.locations()}
data = DataSet(full_dict)
data.to_csv('/home/knut/Downloads/laos_data.csv')


laos_population = '''\
Vientiane Capital: ~820,000
Phongsali: ~177,000
Louangnamtha: ~176,000
Oudomxai: ~307,000
Bokeo: ~205,000
Louangphabang: ~431,000
Houaphan: ~294,000
Xainyabouli: ~381,000
Xiangkhouang: ~252,000
Vientiane: ~432,000
Bolikhamxai: ~275,000
Khammouan: ~415,000
Savannakhet: ~939,000
Salavan: ~396,000
Xekong: ~120,000
Champasak: ~694,000
Attapu: ~153,000
Xaisomboun: ~93,000'''
laos_population = {line.split(': ')[0]: int(line.split(': ~')[1].replace(',', '')) for line in laos_population.split('\n')}
data_dict = {name[3:]: data.data() for name, data in data.items()}
full_data = {name: FullData(d.time_period, d.rainfall, d.mean_temperature, d.disease_cases, np.full(len(d), laos_population[name]))
             for name, d in data_dict.items()}
full_data = DataSet(full_data)
full_data.to_csv('/home/knut/Data/laos_full_data.csv')
#data = {name: FullData.combine(health.get_location(name).data(), spatio_temporal_dict.get_location(name).data(), laos_population[name])
#        for name in health.locations()}
