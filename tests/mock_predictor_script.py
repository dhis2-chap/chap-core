import climate_health
from climate_health.datatypes import ClimateData, ClimateHealthTimeSeries
from climate_health.predictor.naive_predictor import NaivePredictor, MultiRegionNaivePredictor
import typer

from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict

app = typer.Typer()


@app.command()
def predict_values(train_data_set: str, future_climate_data_set: str, output_file: str):
    #train_data = climate_health.datatypes.ClimateHealthTimeSeries.from_csv(train_data_set)
    #future_climate_data = climate_health.datatypes.ClimateData(future_climate_data_set)
    #predictor = NaivePredictor()
    predictor = MultiRegionNaivePredictor()
    train_data = SpatioTemporalDict.from_csv(train_data_set, ClimateHealthTimeSeries)
    future_climate_data = SpatioTemporalDict.from_csv(future_climate_data_set, ClimateData)
    predictor.train(train_data)
    predictions = predictor.predict(future_climate_data)
    print(predictions)
    predictions.to_csv(output_file)


if __name__ == "__main__":
    app()
