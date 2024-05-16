import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from plotly.graph_objs import Figure
import plotly.graph_objects as go

from climate_health.datatypes import ClimateData, HealthData, SummaryStatistics
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


def forecast_plot(true_data: HealthData, predicition_sampler: IsSampler, climate_data: ClimateData,
                  n_samples) -> Figure:
    samples = np.array([predicition_sampler.sample(climate_data) for _ in range(n_samples)])
    quantiles = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)

    return plot_forecast(quantiles, true_data)


def plot_forecast_from_summaries(summaries: SummaryStatistics, true_data: HealthData, transform = lambda x: x) -> Figure:
    df = summaries.topandas()
    true_df = pd.DataFrame({'x': [str(p) for p in true_data.time_period.topandas()], 'real': true_data.disease_cases})
    df.time_period = df.time_period.astype(str)
    return plot_forecasts_from_data_frame(df, true_df, transform)
    # return plot_forecast([summaries.quantile_low, summaries.median, summaries.quantile_high], true_data, x_pred=summaries.time_period.topandas())


def plot_forecast(quantiles: np.ndarray, true_data: HealthData, x_pred=None) -> Figure:
    x_true = [str(p) for p in true_data.time_period.topandas()]

    if x_pred is None:
        x_pred = x_true
    else:
        x_pred = [str(p) for p in x_pred]
    df = pd.DataFrame({'x': x_pred, '10th': quantiles[0], '50th': quantiles[1], '90th': quantiles[2]})
    true_df = pd.DataFrame({'x': x_true, 'real': true_data.disease_cases})
    true_df.x = true_df.x.astype(str)
    return plot_forecasts_from_data_frame(df, true_df)


def plot_forecasts_from_data_frame(prediction_df, true_df, transform = lambda x: x) -> Figure:
    # insert last row of true_df into start of prediction_df
    last_idx = np.where(prediction_df['time_period'][0] == true_df['x'])[0][0]
    if last_idx != 0:
        last_row = true_df.iloc[last_idx-1]
        prepend_df = {'time_period': [last_row['x']], 'quantile_high': last_row['real'], 'quantile_low': last_row['real'], 'median': last_row['real']}
        prediction_df = pd.concat([pd.DataFrame(prepend_df), prediction_df], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prediction_df["time_period"], y=transform(prediction_df["quantile_high"]), mode="lines", line=dict(color='lightgrey'),
                             name="quantile_high"), )
    fig.add_trace(go.Scatter(x=prediction_df["time_period"], y=transform(prediction_df["quantile_low"]),
                             mode="lines", line=dict(color='lightgrey'), fill="tonexty",
                             fillcolor='rgba(68, 68, 68, 0.3)',
                             name="quantile_low"))
    fig.add_scatter(x = true_df['x'], y=transform(true_df['real']), mode='lines', name='real', line=dict(color='blue'))
    fig.add_scatter(x=prediction_df["time_period"], y=transform(prediction_df["median"]), mode="lines", line=dict(color='grey'), name="Median")
    fig.update_layout(
        title="Predicted path using estimated parameters vs real path",
        xaxis_title="Time Period",
        yaxis_title="Disease Cases")
    return fig


def summary_plot(true_data: HealthData, summary_data: SummaryStatistics):
    reporting_rate = 10000
    T = len(true_data) + 1
    for i in range(n_samples):
        new_observed = predicition_sampler.sample(climate_data)
        plt.plot(new_observed, label='predicted', color='grey')
    plt.plot(true_data.disease_cases, label='real', color='blue')
    plt.legend()
    plt.title('Prdicted path using estimated parameters vs real path')
    return plt.gcf()
