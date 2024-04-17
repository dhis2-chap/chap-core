"""Console script for climate_health."""
import json
import webbrowser
from pathlib import PurePath, Path
from typing import Literal

import pandas as pd
from cyclopts import App

from climate_health.dhis2_interface.json_parsing import add_population_data
from climate_health.predictor import get_model, models, ModelType
from climate_health.file_io.example_data_set import datasets, DataSetType
from climate_health.time_period.date_util_wrapper import delta_month
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
    output_filename = f'./{model_name}_{dataset_name}_results.html'
    table_filename = PurePath(output_filename).with_suffix('.csv')
    results.save(output_filename)
    table.to_csv(table_filename)
    webbrowser.open(output_filename)


@app.command()
def forecast(model_name: ModelType, dataset_name: DataSetType, n_months: int):
    logging.basicConfig(level=logging.INFO)
    dataset = datasets[dataset_name].load()
    model = get_model(model_name)()
    do_forecast(model, dataset, n_months * delta_month)

@app.command()
def dhis_pull(base_url: str, username: str, password: str):
    from climate_health.dhis2_interface.ChapProgram import ChapPullPost
    process = ChapPullPost(dhis2Baseurl=base_url.rstrip('/'), dhis2Username=username, dhis2Password=password)

    # set config used in the fetch request
    disease_data_frame = process.pullDHIS2Analytics()
    population_data_frame = process.pullPopulationData()

    path = Path('dhis2analyticsResponses/')
    path.mkdir(exist_ok=True, parents=True)

    disease_filename = (path / process.DHIS2HealthPullConfig.get_id()).with_suffix('.csv')
    population_filename = (path / process.DHIS2PopulationPullConfig.get_id()).with_suffix('.json')
    full_data_frame = add_population_data(disease_data_frame, population_data_frame)
    full_data_frame.to_csv(disease_filename)
    with open(population_filename, 'w') as f:
        json.dump(population_data_frame, f, sort_keys=True, indent=4)


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
