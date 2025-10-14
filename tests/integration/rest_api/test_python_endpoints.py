import pickle
from unittest.mock import patch

import pytest

from chap_core.api_types import PredictionRequest
from chap_core.rest_api.v1.jobs import NaiveJob, NaiveWorker
