"""
Data types for XAI explanations.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from chap_core.database.base_tables import DBModel


class FeatureAttribution(DBModel):
    feature_name: str
    importance: float
    direction: str | None = None
    baseline_value: float | None = None
    actual_value: float | None = None


class GlobalExplanation(BaseModel):
    top_features: list[FeatureAttribution]
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    n_samples: int = 0
    stability_score: float | None = None


class LocalExplanation(BaseModel):
    prediction_id: int
    org_unit: str
    period: str
    output_statistic: str = "median"
    feature_attributions: list[FeatureAttribution]
    baseline_prediction: float
    actual_prediction: float
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
