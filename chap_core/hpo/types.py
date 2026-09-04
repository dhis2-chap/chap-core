from dataclasses import dataclass
from typing import Any

DEFAULT_HPO_TRIALS = 3


# Mainly for future adaptive parallel searching to keep track of trial id/token
@dataclass(frozen=True, slots=True)
class SearchCandidate:
    params: dict[str, Any]
    token: int | None = None
