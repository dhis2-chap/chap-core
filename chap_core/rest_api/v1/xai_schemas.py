from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chap_core.database.base_tables import DBModel
from chap_core.xai.method_registry import SHAP_AUTO
from chap_core.xai.types import FeatureAttribution


class CovariateProvenanceRead(DBModel):
    """Provenance of the covariate row used for a single forecast instance.

    Fields other than ``source`` and ``detail`` are variant-specific:
    only the fields relevant to ``source`` are populated.
    """

    source: Literal[
        "dataset_match",
        "last_available_row",
        "historical_same_month_mean",
        "historical_same_week_mean",
    ]
    detail: str
    matched_period: str | None = None
    aggregate: str | None = None
    target_year: int | None = None
    calendar_month: int | None = None
    iso_week: int | None = None
    n_rows_averaged: int | None = None
    years_used: list[int] | None = None


class SurrogateQualityRead(DBModel):
    r_squared: float | None = None
    mae: float | None = None
    mape: float | None = None
    n_samples: int = 0
    n_unique_rows: int = 0
    constant_features: list[str] = Field(default_factory=list)
    imputation_rates: dict[str, float] = Field(default_factory=dict)
    removed_features: list[str] = Field(default_factory=list)
    selected_model_type: str | None = None
    selected_model_display_name: str | None = None
    n_groups: int | None = None
    fidelity_tier: Literal["good", "moderate", "poor"] | None = None
    residual_mean: float | None = None
    residual_std: float | None = None
    target_transformed: bool = False
    target_transform_method: str | None = None
    permutation_removed_features: list[str] = Field(default_factory=list)
    r_squared_train: float | None = None
    generalization_gap: float | None = None


class GlobalExplanationResponse(DBModel):
    method: str
    top_features: list[FeatureAttribution]
    computed_at: datetime | None = None
    n_samples: int = 0
    stability_score: float | None = None
    surrogate_quality: SurrogateQualityRead | None = None


class LocalExplanationRequest(BaseModel):
    org_unit: str = Field(..., alias="orgUnit")
    period: str
    output_statistic: str = Field("median", alias="outputStatistic")
    xai_method: str = Field(SHAP_AUTO, alias="xaiMethod")
    top_k: int = Field(10, alias="topK")
    force: bool = False

    model_config = ConfigDict(populate_by_name=True)


class LocalExplanationResponse(DBModel):
    id: int | None = None
    prediction_id: int
    org_unit: str
    period: str
    method: str
    output_statistic: str
    feature_attributions: list[FeatureAttribution]
    baseline_prediction: float
    actual_prediction: float
    computed_at: datetime | None = None
    status: str = "completed"
    surrogate_quality: SurrogateQualityRead | None = None
    covariate_provenance: CovariateProvenanceRead | None = None


class RunExplanationsRequest(BaseModel):
    xai_method: str = Field(SHAP_AUTO, alias="xaiMethod")
    output_statistic: str = Field("median", alias="outputStatistic")
    top_k: int = Field(10, alias="topK")

    model_config = ConfigDict(populate_by_name=True)


class ShapBeeswarmPoint(DBModel):
    feature_name: str
    shap_value: float
    feature_value: float
    org_unit: str
    period: str


class ShapBeeswarmResponse(DBModel):
    prediction_id: int
    output_statistic: str
    feature_names: list[str]
    points: list[ShapBeeswarmPoint]
    surrogate_quality: SurrogateQualityRead | None = None


class HorizonFeatureImportance(DBModel):
    feature_name: str
    importance: float
    direction: str


class HorizonStepSummary(DBModel):
    period: str
    target_period: str
    forecast_step: int
    feature_importances: list[HorizonFeatureImportance]
    actual_prediction: float | None = None


class AverageImportance(DBModel):
    feature_name: str
    mean_abs_importance: float
    mean_signed_importance: float
    direction: str


class HorizonSummaryResponse(DBModel):
    prediction_id: int
    org_unit: str
    method: str
    output_statistic: str
    steps: list[HorizonStepSummary]
    average_importance: list[AverageImportance]
    surrogate_quality: SurrogateQualityRead | None = None


class XaiMethodRead(DBModel):
    id: int
    name: str
    display_name: str
    description: str
    method_type: str
    method_type_label: str
    source_url: str | None = None
    author: str
    archived: bool
    is_auto: bool = False
    is_native: bool = False
    supported_visualizations: list[str]
    supported_visualization_labels: list[str]
    default_visualization: str
