"""
NOTE: This file is not used in the current implementation of the CHAP-app. The functionanality is deprecated and this file will
be removed in the future.
"""

import logging


# model_type = Literal[*model_dict.keys()]

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
