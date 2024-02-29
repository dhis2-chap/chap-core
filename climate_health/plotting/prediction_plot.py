import numpy as np
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure

from climate_health.datatypes import ClimateData, HealthData
from climate_health.predictor.protocol import Sampler


def prediction_plot(true_data: HealthData, predicition_sampler: Sampler, climate_data: ClimateData, n_samples)-> Figure:
    reporting_rate = 10000
    T = len(true_data)+1
    for i in range(n_samples):
        new_observed = predicition_sampler.sample(climate_data)
        plt.plot(new_observed, label='predicted', color='grey')
    plt.plot(true_data.disease_cases, label='real', color='blue')
    plt.legend()
    plt.title('Prdicted path using estimated parameters vs real path')
    return plt.gcf()

def forecast_plot(true_data: HealthData, predicition_sampler: Sampler, climate_data: ClimateData, n_samples)-> Figure:
    samples = np.array([predicition_sampler.sample(climate_data) for _ in range(n_samples)])
    quantiles = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
    plt.plot(quantiles[1], label='50th percentile', color='grey')
    x = np.arange(len(true_data.disease_cases))
    plt.fill_between(x, quantiles[0], quantiles[2], color='grey', alpha=0.5)
    plt.plot(quantiles[0], label='10th percentile', color='grey')
    plt.plot(quantiles[2], label='90th percentile', color='grey')
    plt.plot(true_data.disease_cases, label='real', color='blue')
    plt.legend()
    plt.title('Prdicted path using estimated parameters vs real path')
    return plt.gcf()