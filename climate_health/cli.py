"""Console script for climate_health."""
import dataclasses
import json
import webbrowser
from pathlib import PurePath, Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from cyclopts import App
from . import api
from climate_health.dhis2_interface.ChapProgram import ChapPullPost
from climate_health.dhis2_interface.json_parsing import add_population_data, predictions_to_datavalue
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
    out_path = get_results_path() / f'{model_name}_{dataset_name}_forecast_results_{n_months}.html'
    f = open(out_path, "w")
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(prediction.data(),
                                           dataset.get_location(location).data())  # , lambda x: np.log(x+1))
        f.write(fig.to_html())
    f.close()


@app.command()
def dhis_pull(base_url: str, username: str, password: str):
    path = Path('dhis2analyticsResponses/')
    path.mkdir(exist_ok=True, parents=True)
    from climate_health.dhis2_interface.ChapProgram import ChapPullPost
    process = ChapPullPost(dhis2Baseurl=base_url.rstrip('/'), dhis2Username=username, dhis2Password=password)
    full_data_frame = get_full_dataframe(process)
    disease_filename = (path / process.DHIS2HealthPullConfig.get_id()).with_suffix('.csv')
    full_data_frame.to_csv(disease_filename)


def get_full_dataframe(process):
    disease_data_frame = process.pullDHIS2Analytics()
    population_data_frame = process.pullPopulationData()
    full_data_frame = add_population_data(disease_data_frame, population_data_frame)
    return full_data_frame


@app.command()
def dhis_flow(base_url: str, username: str, password: str, n_periods=1):
    process = ChapPullPost(dhis2Baseurl=base_url.rstrip('/'), dhis2Username=username, dhis2Password=password)
    full_data_frame = get_full_dataframe(process)
    modelspec = SSMWithoutWeather()
    model = SSMForecasterNuts(modelspec, NutsParams(n_samples=10, n_warmup=10))
    model.train(full_data_frame)
    predictions = model.prediction_summary(Week(full_data_frame.end_timestamp))
    json_response = process.pushDataToDHIS2(predictions, modelspec.__class__.__name__, do_dict=False)
    # save json
    json_filename = Path('dhis2analyticsResponses/') / f'{modelspec.__class__.__name__}.json'
    with open(json_filename, 'w') as f:
        json.dump(json_response, f, indent=4)


@app.command()
def serve():
    '''
    Start CHAP as a backend server
    '''
    from .rest_api import main_backend
    main_backend()


@dataclasses.dataclass
class AreaPolygons:
    ...


'''@dataclasses.dataclass
class PredictionData:
    area_polygons: AreaPolygons
    health_data: SpatioTemporalDict[HealthData]
    climate_data: SpatioTemporalDict[ClimateData]
    population_data: SpatioTemporalDict[PopulationData]


def read_zip_folder(zip_file_path: str):
    #
    zip_file_reader = ZipFileReader(zip_file_path)
    ...


def convert_geo_json(geo_json_content) -> OurShapeFormat:
    ...

# def write_graph_data(geo_json_content) -> None:
#    ...





# GeoJson convertion
# zip folder reading
# GOthenburg
# Create prediction csv
'''


@app.command()
def dhis_zip_flow(zip_file_path: str, out_json: str, model_name: Optional[str] = None, docker_filename: Optional[str] = None):
    '''
    Run an forecasting evaluation on  data from a zip file from DHIS2, and save the results to a json file
    Run using the specified model_name, which can also be a path to a yaml file. Optionally specify a docker filename

    Parameters:
        zip_file_path: str: Path to the zip file
        out_json: str: Path to the output json file
        model_name: Optional[str]: Name of the model to use, or path to a yaml file
        docker_filename: Optional[str]: Path to a docker file
    '''

    api.dhis_zip_flow(zip_file_path, out_json, model_name, docker_filename=docker_filename)


def main_function():
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function<

    >>> main()

    '''
    return


def main():
    app()


if __name__ == "__main__":
    app()
