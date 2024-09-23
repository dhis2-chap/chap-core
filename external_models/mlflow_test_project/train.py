import sys

print("Training")
sys.exit()


import pickle

import climate_health
from climate_health.datatypes import ClimateData, ClimateHealthTimeSeries
from climate_health.predictor.naive_predictor import NaivePredictor, MultiRegionNaivePredictor
import typer
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet

print("Training")
train_data_set = sys.argv[1]
model_output_file = sys.argv[2]

predictor = MultiRegionNaivePredictor()
train_data = DataSet.from_csv(train_data_set, ClimateHealthTimeSeries)
predictor.train(train_data)

# pickle predictor
with open(model_output_file, 'wb') as f:
    pickle.dump(predictor, f)


