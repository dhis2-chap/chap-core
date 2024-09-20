import databricks.sdk.service.sql
import numpy as np

from climate_health.assessment.forecast import forecast_ahead, forecast_with_predicted_weather
from climate_health.assessment.prediction_evaluator import plot_predictions
from climate_health.data.datasets import ISIMIP_dengue_harmonized
from climate_health.datatypes import FullData, HealthData
from climate_health.external.external_model import get_model_from_directory_or_github_url
from climate_health.external.r_models import models_path
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet

if __name__ == '__main__':
    #model_name = 'https://github.com/sandvelab/chap_auto_ewars'
    model_url = '/home/knut/Sources/chap_auto_ewars_weekly'
    #model_name = models_path / 'naive_python_model_with_mlproject_file'
    estimator = get_model_from_directory_or_github_url(model_url)
    #dataset = ISIMIP_dengue_harmonized['vietnam']
    dataset = DataSet.from_csv('/home/knut/Data/ch_data/weekly_laos_data.csv', FullData)
    predictor = estimator.train(dataset)
    predictions = forecast_with_predicted_weather(predictor, dataset, 26)
    plot_predictions(predictions, dataset, 'prediction_example.pdf')
    medians = DataSet({loc: HealthData(data.time_period, np.median(data.samples, axis=-1)) for loc, data in predictions.items()})
    medians.to_csv('laos_weekly_predictions.csv')

