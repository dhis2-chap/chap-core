import numpy as np

from chap_core.assessment.forecast import forecast_ahead
from chap_core.assessment.prediction_evaluator import plot_predictions
from chap_core.datatypes import FullData, HealthData
from chap_core.external.external_model import get_model_from_directory_or_github_url
from chap_core.predictor.model_registry import registry
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Week
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    #model_url = '/home/knut/Sources/chap_auto_ewars_weekly'
    model_name = 'auto_regressive_weekly'# chap_ewars_weekly'
    estimator= registry.get_model(model_name)
    #estimator = get_model_from_directory_or_github_url(model_url)
    full_dataset = DataSet.from_csv('/home/knut/Data/ch_data/weekly_laos_data.csv', FullData)
    for split_point in [Week(2024, 1),
                        Week(2024, 13),
                        Week(2024, 25),
                        Week(2024, 37)]:
        logger.info(f'Running split {split_point} for model {model_name}')
        dataset = full_dataset.restrict_time_period(slice(None, split_point))
        logger.info(f'Running forecast ahead for split {split_point} for model {model_name}')
        predictions = forecast_ahead(estimator, dataset, 12)
        logger.info(f'Running plot predictions for split {split_point} for model {model_name}')
        plot_predictions(predictions, dataset, f'laos_weekly_predictions_split{split_point.id}_{model_name}.pdf')
        logger.info(f'Running medians for split {split_point} for model {model_name}')
        medians = DataSet({loc: HealthData(data.time_period, np.median(data.samples, axis=-1)) for loc, data in predictions.items()})
        medians.to_csv(f'laos_weekly_predictions_split{split_point.id}_{model_name}.csv')