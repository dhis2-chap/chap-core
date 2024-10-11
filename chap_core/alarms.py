from typing import Iterable

import numpy as np
from pydantic import BaseModel

# Epidemioglical week: different from calendar week
# https://www.cdc.gov/flu/weekly/overview.htm


class OutbreakParameters(BaseModel):
    endemic_factor: float
    probability_threshold: float


def outbreak_prediction(parameters: OutbreakParameters, case_samples: Iterable[float]) -> bool:
    return np.mean()
