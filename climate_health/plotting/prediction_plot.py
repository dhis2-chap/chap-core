import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from plotly.graph_objs import Figure
import plotly.graph_objects as go

from climate_health.datatypes import ClimateData, HealthData
from climate_health.predictor.protocol import IsSampler


def prediction_plot(true_data: HealthData, predicition_sampler: IsSampler, climate_data: ClimateData,
                    n_samples) -> Figure:
    reporting_rate = 10000
    T = len(true_data) + 1
    for i in range(n_samples):
        new_observed = predicition_sampler.sample(climate_data)
        plt.plot(new_observed, label='predicted', color='grey')
    plt.plot(true_data.disease_cases, label='real', color='blue')
    plt.legend()
    plt.title('Prdicted path using estimated parameters vs real path')
    return plt.gcf()


def forecast_plot(true_data: HealthData, predicition_sampler: IsSampler, climate_data: ClimateData, n_samples) -> Figure:
    samples = np.array([predicition_sampler.sample(climate_data) for _ in range(n_samples)])
    quantiles = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
    x = ["-".join([str(x.year), str(x.month+1).zfill(2)]) for x in true_data.time_period]
    df = pd.DataFrame({'x': x, '10th': quantiles[0], '50th': quantiles[1], '90th': quantiles[2]})

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["x"], y=df["10th"], mode="lines", line=dict(color='lightgrey'), fill=None, name="10th percentile"))
    fig.add_trace(go.Scatter(x=df["x"], y=df["50th"], mode="lines", line=dict(color='grey'), fill="tonexty", name="50th percentile"))
    fig.add_trace(go.Scatter(x=df["x"], y=df["90th"], mode="lines", line=dict(color='black'), fill="tonexty", name="90th percentile"))

    fig.add_scatter(x=x, y=true_data.disease_cases, mode='lines', name='real', line=dict(color='blue'))

    fig.update_layout(
        title="Predicted path using estimated parameters vs real path",
        xaxis_title="Time Period",
        yaxis_title="Disease Cases")

    return fig

