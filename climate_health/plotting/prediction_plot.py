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
