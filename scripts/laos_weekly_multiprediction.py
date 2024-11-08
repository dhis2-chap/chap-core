import numpy as np

from chap_core.assessment.forecast import forecast_ahead
from chap_core.assessment.prediction_evaluator import plot_predictions
from chap_core.datatypes import FullData, HealthData
from chap_core.external.external_model import get_model_from_directory_or_github_url
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Week


if __name__ == '__main__':
    model_url = '/home/knut/Sources/chap_auto_ewars_weekly'
    estimator = get_model_from_directory_or_github_url(model_url)
    full_dataset = DataSet.from_csv('/home/knut/Data/ch_data/weekly_laos_data.csv', FullData)
    for split_point in [Week(2024, 1),
                        Week(2024, 13),
                        Week(2024, 25),
                        Week(2024, 37)]:
        dataset = full_dataset.restrict_time_period(slice(None, split_point))
        predictions = forecast_ahead(estimator, dataset, 12)
        plot_predictions(predictions, dataset, f'laos_weekly_predictions_split{split_point.id}.pdf')
        medians = DataSet({loc: HealthData(data.time_period, np.median(data.samples, axis=-1)) for loc, data in predictions.items()})
        medians.to_csv(f'laos_weekly_predictions_split{split_point.id}.csv')

