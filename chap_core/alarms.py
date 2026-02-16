from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel

# Epidemioglical week: different from calendar week
# https://www.cdc.gov/flu/weekly/overview.htm


class OutbreakParameters(BaseModel):
    endemic_factor: float
    probability_threshold: float


def outbreak_prediction(parameters: OutbreakParameters, case_samples: Iterable[float]) -> bool:
    return bool(np.mean(list(case_samples)) > parameters.endemic_factor * parameters.probability_threshold)
