"""Console script for climate_health."""
import json
import webbrowser
from pathlib import PurePath, Path
from typing import Literal

import pandas as pd
from cyclopts import App

from climate_health.dhis2_interface.ChapProgram import ChapPullPost
from climate_health.dhis2_interface.json_parsing import add_population_data, predictions_to_json
from climate_health.external.models.jax_models.model_spec import SSMForecasterNuts, NutsParams
from climate_health.external.models.jax_models.specs import SSMWithoutWeather, NaiveSSM
from climate_health.file_io import get_results_path
from climate_health.plotting.prediction_plot import plot_forecast_from_summaries
from climate_health.predictor import get_model, models, ModelType
from climate_health.file_io.example_data_set import datasets, DataSetType
from climate_health.time_period.date_util_wrapper import delta_month, Week
from .assessment.prediction_evaluator import evaluate_model
from .assessment.forecast import forecast as do_forecast
import logging

app = App()


def append_to_csv(file_object, data_frame: pd.DataFrame):
    data_frame.to_csv(file_object, mode='a', header=False)


@app.command()
def evaluate(model_name: ModelType, dataset_name: DataSetType, max_splits: int, other_model: ModelType = None):
    '''
    Evaluate a model on a dataset using forecast cross validation
    '''
    logging.basicConfig(level=logging.INFO)
    dataset = datasets[dataset_name].load()
    model = get_model(model_name)()
    f = open('debug.csv', 'w')
    callback = lambda name, data: append_to_csv(f, data.to_pandas())
    results, table = evaluate_model(dataset, model, max_splits, start_offset=24, return_table=True,
                                    naive_model_cls=get_model(other_model) if other_model else None, callback=callback,
                                    mode='prediction_summary')
    output_filename = get_results_path() / f'{model_name}_{dataset_name}_results.html'
    table_filename = PurePath(output_filename).with_suffix('.csv')
    results.save(output_filename)
    table.to_csv(table_filename)
    webbrowser.open(output_filename)


@app.command()
def forecast(model_name: ModelType, dataset_name: DataSetType, n_months: int):
    logging.basicConfig(level=logging.INFO)
    dataset = datasets[dataset_name].load()
    model = get_model(model_name)()
    predictions = do_forecast(model, dataset, n_months * delta_month)
    out_path = get_results_path()/f'{model_name}_{dataset_name}_forecast_results_{n_months}.html'
    f = open(out_path, "w")
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(prediction.data(), dataset.get_location(location).data())
        f.write(fig.to_html())
    f.close()

@app.command()
def dhis_pull(base_url: str, username: str, password: str):
    from climate_health.dhis2_interface.ChapProgram import ChapPullPost
    

    full_data_frame, population_data_frame = get_full_dataframe(process)

    path = Path('dhis2analyticsResponses/')
    path.mkdir(exist_ok=True, parents=True)

    # set config used in the fetch requestprocess = ChapPullPost(dhis2Baseurl=base_url.rstrip('/'), dhis2Username=username, dhis2Password=password)
    process = ChapPullPost(dhis2Baseurl=base_url.rstrip('/'), dhis2Username=username, dhis2Password=password)
    full_data_frame = get_full_dataframe(process)
    disease_filename = (path / process.DHIS2HealthPullConfig.get_id()).with_suffix('.csv')
    full_data_frame.to_csv(disease_filename)


def get_full_dataframe(process):

    # set config used in the fetch request
    disease_data_frame = process.pullDHIS2Analytics()
    population_data_frame = process.pullPopulationData()
    full_data_frame = add_population_data(disease_data_frame, population_data_frame)
    return full_data_frame


@app.command()
def dhis_flow(base_url: str, username: str, password: str, n_periods = 1):
    process = ChapPullPost(dhis2Baseurl=base_url.rstrip('/'), dhis2Username=username, dhis2Password=password)
    full_data_frame = get_full_dataframe(process)
    model = SSMForecasterNuts(SSMWithoutWeather(), NutsParams(n_samples=10, n_warmup=10))
    model.train(full_data_frame)
    predictions = model.prediction_summary(Week(full_data_frame.end_timestamp))
    json = predictions_to_json(predictions)
    print(json)



def main_function():
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main()

    '''
    return


def main():
    app()


if __name__ == "__main__":
    app()
