from gluonts.model.estimator import Estimator
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from ..data import DataSet
from ..data.gluonts_adaptor.dataset import DataSetAdaptor
from ..datatypes import Samples
from ..time_period import PeriodRange
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GluonTSPredictor:
    gluonts_predictor: Predictor

    def predict(self, history: DataSet, future_data: DataSet, num_samples=100) -> DataSet:
        gluonts_dataset = DataSetAdaptor.to_gluonts_testinstances(
            history, future_data, self.gluonts_predictor.prediction_length
        )
        forecasts = self.gluonts_predictor.predict(gluonts_dataset, num_samples=num_samples)
        data = {
            location: Samples(PeriodRange.from_pandas(forecast.index), forecast.samples.T)
            for location, forecast in zip(history.keys(), forecasts)
        }
        return DataSet(data)

    def save(self, filename: str):
        filepath = Path(filename)
        filepath.mkdir(exist_ok=True, parents=True)
        self.gluonts_predictor.serialize(filepath)

    @classmethod
    def load(cls, filename: str):
        return GluonTSPredictor(Predictor.deserialize(Path(filename)))


@dataclass
class GluonTSEstimator:
    gluont_ts_estimator: Estimator

    def train(self, dataset: DataSet) -> GluonTSPredictor:
        gluonts_dataset = DataSetAdaptor.to_gluonts(dataset)
        ds = ListDataset(gluonts_dataset, freq="m")
        return GluonTSPredictor(self.gluont_ts_estimator.train(ds))
