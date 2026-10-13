"""Console script for ch_modelling."""
import os

from chap_core.datatypes import FullData, remove_field
from cyclopts import App
from chap_core.data import DataSet
from chap_core.predictor.naive_estimator import NaiveEstimator, NaivePredictor

app = App()


@app.command()
def train(training_data_filename: str, model_path: str):
    '''
    '''
    assert False
    dataset = DataSet.from_csv(training_data_filename, FullData)
    chap_estimator = NaiveEstimator
    predictor = chap_estimator().train(dataset)
    print('cur dir', os.getcwd())
    print("Saving model to", model_path)
    predictor.save(model_path)


@app.command()
def predict(model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str):
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function
    '''
    dataset = DataSet.from_csv(historic_data_filename, FullData)
    future_data = DataSet.from_csv(future_data_filename, remove_field(FullData, 'disease_cases'))
    chap_predictor = NaivePredictor
    predictor = chap_predictor.load(model_filename)
    forecasts = predictor.predict(dataset, future_data)
    forecasts.to_csv(output_filename)


def main():
    app()


if __name__ == "__main__":
    app()
