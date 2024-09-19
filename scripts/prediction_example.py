from climate_health.assessment.forecast import forecast_ahead, forecast_with_predicted_weather
from climate_health.assessment.prediction_evaluator import plot_predictions
from climate_health.data.datasets import ISIMIP_dengue_harmonized
from climate_health.external.external_model import get_model_from_directory_or_github_url
from climate_health.external.r_models import models_path

if __name__ == '__main__':
    model_name = 'https://github.com/sandvelab/chap_auto_ewars'
    #model_name = models_path / 'naive_python_model_with_mlproject_file'
    estimator = get_model_from_directory_or_github_url(model_name)
    dataset = ISIMIP_dengue_harmonized['vietnam']
    predictor = estimator.train(dataset)
    predictions = forecast_with_predicted_weather(predictor, dataset, 3)
    plot_predictions(predictions, dataset, 'prediction_example.pdf')

