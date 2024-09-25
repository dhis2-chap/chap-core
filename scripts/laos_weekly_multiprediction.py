import databricks.sdk.service.sql
import numpy as np

from climate_health.assessment.forecast import forecast_ahead
from climate_health.assessment.prediction_evaluator import plot_predictions
from climate_health.datatypes import FullData, HealthData
from climate_health.external.external_model import get_model_from_directory_or_github_url
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet
from climate_health.time_period import Week


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

