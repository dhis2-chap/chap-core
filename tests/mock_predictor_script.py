import pickle

from chap_core.datatypes import (
    ClimateData,
    ClimateHealthTimeSeries,
    SimpleClimateData,
)
from chap_core.predictor.naive_predictor import (
    MultiRegionNaivePredictor,
)
import typer

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

app = typer.Typer()


@app.command()
def train(train_data_set: str, model_output_file: str):
    predictor = MultiRegionNaivePredictor()
    train_data = DataSet.from_csv(train_data_set, ClimateHealthTimeSeries)
    predictor.train(train_data)

    # pickle predictor
    with open(model_output_file, "wb") as f:
        pickle.dump(predictor, f)


@app.command()
def predict(future_climate_data_set: str, model_file: str, output_file: str):
    with open(model_file, "rb") as f:
        predictor = pickle.load(f)

    future_climate_data = DataSet.from_csv(future_climate_data_set, SimpleClimateData)
    predictions = predictor.predict(future_climate_data)
    print(predictions)
    predictions.to_csv(output_file)


@app.command()
def predict_values(train_data_set: str, future_climate_data_set: str, output_file: str):
    predictor = MultiRegionNaivePredictor()
    train_data = DataSet.from_csv(train_data_set, ClimateHealthTimeSeries)
    future_climate_data = DataSet.from_csv(future_climate_data_set, ClimateData)
    predictor.train(train_data)
    predictions = predictor.predict(future_climate_data)
    print(predictions)
    predictions.to_csv(output_file)


if __name__ == "__main__":
    app()
