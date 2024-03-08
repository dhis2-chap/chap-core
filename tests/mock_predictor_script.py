import climate_health
from climate_health.predictor.naive_predictor import NaivePredictor
import typer

app = typer.Typer()

@app.command()
def predict_values(train_data_set: str, future_climate_data_set: str, output_file: str):
    train_data = climate_health.datatypes.ClimateHealthTimeSeries.from_csv(train_data_set)
    future_climate_data = climate_health.datatypes.ClimateData(future_climate_data_set)
    predictor = NaivePredictor()
    predictor.train(train_data)
    predictions = predictor.predict(future_climate_data)
    predictions.to_csv(output_file, index=False)


if __name__ == "__main__":
    app()
