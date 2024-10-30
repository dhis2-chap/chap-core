from chap_core.data.open_dengue import OpenDengueDataSet
from chap_core.climate_data.harmonization import harmonize_health_dataset
from chap_core.datatypes import HealthData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import matplotlib.pyplot as plt
import pandas as pd
if False:
    vietnam_data = OpenDengueDataSet().subset('Viet nam', spatial_resolution='Admin1', temporal_resolution='Month')
    vietnam_data.to_csv('vietnam_monthly.csv', index=False)
else:
    vietnam_data = pd.read_csv('vietnam_monthly.csv')
print(vietnam_data.columns)

def downsample_months_to_weeks(df):
    downsampled = []
    for name, group in df.groupby('adm_1_name'):
        period_index = pd.PeriodIndex(group['calendar_start_date'], freq='M')
        group = group.set_index(period_index)
        days_in_month = group.index.days_in_month
        group['dengue_total'] = group['dengue_total'] / days_in_month * 7
        week_group = group['dengue_total'].resample('W').interpolate()
        new_group = pd.DataFrame({'location': name, 'time_period': week_group.index,
                                  'disease_cases': week_group.values})
        downsampled.append(new_group)
    return pd.concat(downsampled)

if __name__ == '__main__':
    downsampled = downsample_months_to_weeks(vietnam_data)
    dataset = DataSet.from_pandas(downsampled, dataclass=HealthData, fill_missing=True)
    harmonized = harmonize_health_dataset(dataset, 'viet nam', get_climate=False)
    harmonized.to_csv('vietnam_downsampled_weekly.csv')
    downsampled.plot()
    print(downsampled)
    plt.show()
    exit()

'''
for name, group in vietnam_data.groupby('adm_1_name'):
    period_index = pd.PeriodIndex(group['calendar_start_date'], freq='M')
    group = group.set_index(period_index)
    print(group.iloc[:4]['dengue_total'])
    group['dengue_total'].plot()
    plt.show()
    days_in_month = group.index.days_in_month
    group['dengue_total'] = group['dengue_total']/days_in_month*7
    week_group = group['dengue_total'].resample('W').interpolate()
    week_group.plot()
    print(week_group.iloc[:4])
    plt.show()
    break
'''