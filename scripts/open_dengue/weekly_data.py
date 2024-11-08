import pandas as pd
from matplotlib import pyplot as plt

filepath = '~/Downloads/Temporal_extract_V1_2_2.csv'
df: pd.DataFrame = pd.read_csv(filepath)
weekly_mask = df['T_res'] == 'Week'
df = df[weekly_mask]
for admin_level_n in [1, 2]:
    admin_level = 'Admin{}'.format(admin_level_n)
    _df = df[df['S_res'] == admin_level]
    print(_df['adm_0_name'].value_counts())

#df = df[weekly_mask]
if False:
    country_name = 'BRAZIL'
    small_data = df[df['adm_0_name'] == country_name]
    for name, group in small_data.groupby('adm_1_name'):
        group['dengue_total'].plot()
        plt.title(f'{name}_{group.iloc[-1]["calendar_end_date"]}')
        plt.show()

