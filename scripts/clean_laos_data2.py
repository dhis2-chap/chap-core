import string

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from chap_core.datatypes import ClimateData, ClimateHealthTimeSeries, FullData, ClimateHealthData
from chap_core.file_io.cleaners import laos_data
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import TimePeriod

filname = '/home/knut/Downloads/laodenguedata.csv'


def parse_week(week):
    week, year = week.split()
    print(week, year)
    weekstr = string.Formatter().format_field(int(week[1:]), '02')
    return f'{year}W{weekstr}'


raw_df = pd.read_csv(filname)
#
raw_df['Location'] = raw_df['Organisation unit']
# Make a column for each unique value in the 'Data' column
df = raw_df.pivot(index=['Period', 'Location'], columns='Data', values='Value')
df = df.reset_index()
df['Period'] = [parse_week(week) for week in df.Period]
colnames = ['Climate-Rainfall', 'Climate-Temperature avg',
            'NCLE: 7. Dengue cases (any)', 'Location', 'Period']
true_colnames = ['rainfall', 'mean_temperature', 'disease_cases', 'location', 'time_period']
df.rename(columns={colname: true_colname for colname, true_colname in zip(colnames, true_colnames)}, inplace=True)
df = df.sort_values(by=['time_period', 'location'])


def add_population_data(data):
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
    laos_population = {line.split(': ')[0].strip(): int(line.split(': ~')[1].replace(',', '')) for line in
                       laos_population.split('\n')}
    data_dict = {name[3:]: data for name, data in data.items()}
    for name in data_dict.keys():
        if name not in laos_population:
            print(f'{name} not in population data', laos_population.keys())
    full_data = {name: FullData(d.time_period, d.rainfall, d.mean_temperature, d.disease_cases,
                                np.full(len(d), laos_population[name]))
                 for name, d in data_dict.items()}
    return DataSet(full_data)


if __name__ == '__main__':
    dataset = DataSet.from_pandas(
        df,
        dataclass=ClimateHealthData,
        fill_missing=True)
    dataset = add_population_data(dataset)
    dataset.to_csv('/home/knut/Data/ch_data/weekly_laos_data.csv')



if False:
    mapping = {'rainfall': 'gsiW9SgolNd',
               'mean_temperature': 'VA05qvanuVs',
               'max_temperature': 'ZH76qVQl5Mz'}

    health_filaname = '/home/knut/Downloads/dengue.csv'
    df = laos_data(health_filaname)

    # df.to_csv('/home/knut/Downloads/dengue_clean.csv')
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
    full_dict = {name: ClimateHealthTimeSeries.combine(health.get_location(name).data(),
                                                       spatio_temporal_dict.get_location(name).data())
                 for name in health.locations()}
    data = DataSet(full_dict)
    data.to_csv('/home/knut/Downloads/laos_data.csv')

    full_data = DataSet(full_data)
    full_data.to_csv('/home/knut/Data/laos_full_data.csv')
    # data = {name: FullData.combine(health.get_location(name).data(), spatio_temporal_dict.get_location(name).data(), laos_population[name])
    #        for name in health.locations()}
