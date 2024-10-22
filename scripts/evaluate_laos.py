from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.external.external_model import get_model_from_directory_or_github_url
from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
#model_url = '/home/knut/Sources/chap_auto_ewars_weekly'
#model_url = 'https://github.com/sandvelab/chap_auto_ewars'
model_url = 'https://github.com/sandvelab/chap_auto_ewars_weekly'
model = get_model_from_directory_or_github_url(model_url)
dataset = DataSet.from_csv('/home/knut/Data/ch_data/weekly_laos_data.csv', FullData)
if __name__ == '__main__':
    evaluate_model(model, dataset, prediction_length=12, n_test_sets=41, report_filename='laos_weekly_report_2.pdf',
                   weather_provider=QuickForecastFetcher)
