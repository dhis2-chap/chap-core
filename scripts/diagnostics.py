import pandas as pd
import plotly.express as px
import numpy as np

dataset = pd.read_csv('../example_data/hydromet_clean.csv')
dataset.sort_values(by='location')
i = 0
months = []
log_ratios = []
temperature = []
for name, group in dataset.groupby('location'):
    i+=1
    if i>20:
        break
    months.append([int(t.split('-')[-1]) for t in group['time_period'].values[:-1]])
    ratio = (group['disease_cases'][1:].values+1)/(group['disease_cases'][:-1].values+1)
    log_ratios.append(np.log(ratio))
    temperature.append(group['mean_temperature'].values[:-1])

df = pd.DataFrame({'month': np.concatenate(months), 'log_ratio': np.concatenate(log_ratios), 'temperature': np.concatenate(temperature)})
for month in range(1, 13):
    px.scatter(df[df['month']==month], x='temperature', y='log_ratio').show()
#px.scatter(x=group['mean_temperature'].values[:-1], y=np.log(ratio), title=f'Ratio of disease cases for {name}').show()
