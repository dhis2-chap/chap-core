import numpy as np
import pandas as pd
import plotly.express as px
import scipy
import plotly.figure_factory as ff

dataset = pd.read_csv('../example_data/hydromet_clean.csv')
values = []
for i, (name, group) in enumerate(dataset.groupby('location')):
    if i > 10:
        break
    log_cases = np.log(np.array(group['disease_cases'])[12:]+1)
    print(np.array(group['disease_cases'][12:]))
    print(log_cases)

    y = log_cases[1:] - log_cases[:-1]

    y = y[~np.isnan(y) & ~np.isinf(y)]
    if len(y) > 1:
        values.append(y/np.std(y))
    #px.histogram(y).show()

values = np.concatenate(values)
fig = px.histogram(values, nbins=100, title='Histogram of log ratios')
ff.create_distplot([values], group_labels=['log ratios'], show_rug=False).show()

# x = np.linspace(np.min(values), np.max(values), 100)
# y = scipy.stats.norm.pdf(x, 0, np.std(values))
# fig.add_scatter(x = x, y = y*sum(values)*10, mode='lines')
# fig.show()