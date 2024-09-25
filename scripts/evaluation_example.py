# This is an example of how you can use the gluonts dataset adaptor to evaluate models on a dataset using gluonts
# Parts of this functinality will be simplified into CHAP in the future
# For this tutorial  you will need to install gluonts
# pip install gluonts[torch]
import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import period_index
from gluonts.evaluation import Evaluator
from matplotlib import pyplot as plt

from chap_core.data import adaptors, datasets

# Load the dataset and extract the data for Laos
dataset = datasets.ISIMIP_dengue_harmonized
laos_data = dataset['brazil']
n_locations = len(laos_data.locations())
# Convert the dataset to a gluonts dataset
from gluonts.dataset.common import ListDataset

dataset = adaptors.gluonts.from_dataset(laos_data)
dataset = ListDataset(dataset, freq='M')

# Define a simple DeepAR model
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput

predict_length = 4  # 4 months
testing_length = 12  # 12 months

estimator = DeepAREstimator(
    num_layers=2,
    hidden_size=12,
    weight_decay=1e-4,
    dropout_rate=0.2,
    num_feat_static_cat=1,
    embedding_dimension=[1],
    cardinality=[n_locations],
    prediction_length=4,
    freq='M',
    distr_output=NegativeBinomialOutput(),  # For count data
    trainer_kwargs={
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "max_epochs": 20,
    },
)

# Split the dataset into training and test set
from gluonts.dataset.split import split

train, test = split(dataset, offset=testing_length)
predictor = estimator.train(train)

test_instances = test.generate_instances(prediction_length=predict_length, windows=testing_length - predict_length + 1,
                                         distance=1)

def _to_dataframe(input_label):
    """
    Turn a pair of consecutive (in time) data entries into a dataframe.
    """
    start = input_label[0][FieldName.START]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)

def evaluate_on_split(predictor, test_instances):
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    forecast_it = predictor.predict(test_instances.input, num_samples=100)
    ts_it = map(_to_dataframe, test_instances)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    for forecast_entry, ts_entry in zip(forecasts, tss):
        plt.plot(ts_entry[-150:].to_timestamp())
        forecast_entry.plot(show_label=True)
        plt.legend()
        plt.show()
    return agg_metrics

metrics = evaluate_on_split(predictor, test_instances)
