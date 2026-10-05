"""Adapt a selected dataset target to CHAP's internal truth field and a model's input."""

from chap_core.assessment.prediction_evaluator import Estimator, Predictor
from chap_core.datatypes import create_tsdataclass
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def rename_target(dataset: DataSet, source: str, target: str) -> DataSet:
    if source == target:
        return dataset
    fields = {name: name for name in dataset.field_names() if name not in (source, target)}
    fields[target] = source
    data_class = create_tsdataclass(list(fields))
    return DataSet(
        {
            location: data_class(data.time_period, **{name: getattr(data, old) for name, old in fields.items()})
            for location, data in dataset.items()
        },
        polygons=dataset.polygons,
        metadata=dataset.metadata,
    )


class TargetColumnEstimator:
    """Keep CHAP's truth/masking convention while passing the target name a model expects."""

    def __init__(self, estimator: Estimator, target: str):
        self.estimator = estimator
        self.target = target
        self.predictor: Predictor | None = None

    def train(self, data: DataSet):
        self.predictor = self.estimator.train(rename_target(data, "disease_cases", self.target))
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet):
        assert self.predictor is not None
        return self.predictor.predict(rename_target(historic_data, "disease_cases", self.target), future_data)
