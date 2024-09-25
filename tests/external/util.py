import numpy as np

from chap_core.datatypes import FullData
from chap_core.plotting.prediction_plot import plot_forecast_from_summaries
from chap_core.time_period import delta_month


def check_model(full_train_data, model, random_key, test_data):
    true_data, test_data = test_data
    train_data = full_train_data
    for key, value in train_data.items():
        ...  # px.line(y=value.data().disease_cases).show()
    test_data = test_data.remove_field("max_temperature")
    test_data = test_data.add_fields(
        FullData,
        population=lambda data: [100000] * len(data),
        disease_cases=lambda data: [np.nan] * len(data),
    )

    model.train(train_data)
    model.diagnose()
    # results = model.sample(test_data)
    predictions = model.forecast(
        test_data, n_samples=100, forecast_delta=12 * delta_month
    )
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(
            prediction.data(),
            true_data.get_location(location).data(),
            lambda x: np.log(x + 1),
        )
        fig.show()
