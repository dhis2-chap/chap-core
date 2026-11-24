from sqlalchemy import JSON, Column
from sqlmodel import Field

from chap_core.database.base_tables import DBModel


class HpoRunDB(DBModel, table=True):
    """Persisted metadata for a completed HPO run."""

    id: int | None = Field(default=None, primary_key=True)
    # Optimization settings
    searcher: str
    metric: str
    direction: str
    search_space: dict = Field(sa_column=Column(JSON))
    max_trials: int | None = None
    seed: int | None = None
    # Optimization results
    best_params: dict = Field(sa_column=Column(JSON))
    best_score: float
    n_trials: int
    n_successful_trials: int
    n_failed_trials: int
    elapsed_seconds: float
    stop_reason: str
    leaderboard: list[dict] = Field(
        default_factory=list,
        sa_column=Column(JSON),
    )
