from typing import Optional

from chap_core.assessment.prediction_evaluator import Estimator
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import delta_year


def validate_training_data(dataset: DataSet, estimator: Optional[Estimator]) -> None:
    """
    Validate the training data
    """
    assert isinstance(dataset, DataSet)
    if dataset.end_timestamp < dataset.start_timestamp+ 2*delta_year:
        raise ValueError("Training data must cover at least two whole years")