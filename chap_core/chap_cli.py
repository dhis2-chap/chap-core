"""
NOTE: This file is not used in the current implementation of the CHAP-app. The functionanality is deprecated and this file will
be removed in the future.
"""

import json
import logging
from pathlib import Path

from cyclopts import App

from chap_core.api_types import PredictionRequest
from chap_core.assessment.forecast import forecast_ahead
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.datatypes import FullData
from chap_core.geoutils import buffer_point_features, inspect_feature_collection

# model_type = Literal[*model_dict.keys()]
from chap_core.rest_api.worker_functions import dataset_from_request_v1
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import delta_month

logger = logging.getLogger(__name__)


# def _get_model(model_id: str):
#     if model_id == "naive_model":
#         return NaiveEstimator()
#     elif model_id == "chap_ewars_monthly":
#         return get_model_from_directory_or_github_url(
#             "https://github.com/sandvelab/chap_auto_ewars"
#         )
#     elif model_id == "chap_ewars_weekly":
#         return get_model_from_directory_or_github_url(
#             "https://github.com/sandvelab/chap_auto_ewars_weekly")
#     else:
#         raise ValueError(
#             f"Unknown model id: {model_id}, expected one of 'naive_model', 'chap_ewars'"
#         )
