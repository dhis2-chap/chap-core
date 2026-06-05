from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from chap_core.api_types import BacktestParams, FeatureCollectionModel
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_tables import DataSetCreateInfo, ObservationBase
from chap_core.database.model_templates_and_config_tables import (
    ModelTemplateInformation,
    ModelTemplateMetaData,
)
from chap_core.database.tables import BacktestBase, BacktestForecast, BacktestMetric, BacktestRead, QuantileTarget


class PredictionBase(BaseModel):
    """Shared identifying triple for a single predicted value (org unit + period + which element)."""

    orgUnit: str = Field(description="Org-unit identifier the value is for.")
    dataElement: str = Field(description="External data element id the value belongs to.")
    period: str = Field(description="Period the value is for.")


class PredictionResponse(PredictionBase):
    """`PredictionBase` plus a single scalar value (used for non-sample-based predictions)."""

    value: float = Field(description="Predicted value.")


class PredictionSamplResponse(PredictionBase):
    """`PredictionBase` plus the posterior sample values (caller derives quantiles client-side)."""

    values: list[float] = Field(description="Posterior sample values for this (org-unit, period, element).")


class FullPredictionResponse(BaseModel):
    """Aggregate response carrying every predicted value for one disease."""

    diseaseId: str = Field(description="Identifier of the disease the predictions are for.")
    dataValues: list[PredictionResponse] = Field(description="One row per (org-unit, period, element).")


class FetchRequest(DBModel):
    """Tell the server which feature to fetch from which data source when materialising a dataset."""

    feature_name: str = Field(description="Canonical feature name (matching a `FeatureType.name`).")
    data_source_name: str = Field(description="Canonical name of the source to pull from.")


class DatasetMakeRequest(DataSetCreateInfo):
    """Request body for building a dataset: metadata + polygons + observations (provided or fetched)."""

    geojson: FeatureCollectionModel = Field(description="GeoJSON polygon set for the dataset's org units.")
    provided_data: list[ObservationBase] = Field(description="Observations the caller is supplying directly.")
    data_to_be_fetched: list[FetchRequest] = Field(
        description="Features whose observations the server should fetch from a registered source."
    )


class JobResponse(BaseModel):
    """Response returned from any endpoint that enqueues background work."""

    id: str = Field(description="Identifier of the queued job; use it to poll status via the jobs endpoints.")


class PredictionParams(DBModel):
    """Shared prediction-side parameters embedded in request models."""

    model_id: str = Field(description="Canonical name of the configured model to run.")
    n_periods: int = Field(default=3, gt=0, description="Number of future periods to forecast.")


class ValidationError(DBModel):
    """One row of structured validation feedback returned from CSV/JSON dataset imports."""

    reason: str = Field(description="Human-readable reason the row was rejected.")
    org_unit: str = Field(description="Org-unit identifier the rejected row referred to.")
    feature_name: str = Field(description="Canonical feature name the rejected row referred to.")
    time_periods: list[str] = Field(description="Affected periods (may be a single period or a range).")


class ImportSummaryResponse(DBModel):
    """Result of a dataset import: how many rows landed, how many were rejected, and why."""

    id: str | None = Field(
        description="Identifier of the imported dataset; `None` if the import was rejected outright."
    )
    imported_count: int = Field(description="Number of observations that were successfully imported.")
    rejected: list[ValidationError] = Field(description="One row per rejected observation, with the reason.")


class BacktestCreate(BacktestBase):
    """Request body for creating a backtest row directly (DB-level — typically the long path goes via `MakeBacktestRequest`)."""

    # Accept either the configured-model integer primary key or its string
    # name. The underlying DB column (`BacktestBase.model_id`) is a string,
    # but the `POST /v1/crud/backtests/` handler resolves an `int` to the
    # corresponding name before persisting so the column stays consistent.
    # See `chap_core.rest_api.v1.routers.crud.create_backtest` for the
    # resolution path.
    model_id: int | str = Field(  # type: ignore[assignment]
        description="Configured model to backtest: either the integer primary key or the canonical string name.",
    )


class BacktestFull(BacktestRead):
    """Full backtest read view: `BacktestRead` plus every per-(period, org-unit) metric and forecast."""

    metrics: list[BacktestMetric] = Field(description="Per-(period, org-unit) metric values for this backtest.")
    forecasts: list[BacktestForecast] = Field(description="Per-(period, org-unit) forecast rows for this backtest.")


class BacktestDomain(DBModel):
    """The set of org units + split periods a backtest covers — used by the UI to filter visualisations."""

    org_units: list[str] = Field(description="Identifiers of every org unit the backtest scored predictions over.")
    split_periods: list[str] = Field(
        description="Periods at which the rolling backtest's train/test split was advanced."
    )


class ChapDataSource(DBModel):
    """Catalogue entry describing one registered data source (chap-side metadata, not the DB-table row)."""

    name: str = Field(description="Canonical identifier of the source.")
    display_name: str = Field(description="Human-friendly name shown in source pickers.")
    supported_features: list[str] = Field(description="Canonical feature names this source can deliver.")
    description: str = Field(description="Short paragraph describing what this source provides.")
    dataset: str = Field(description="Canonical name of the upstream dataset this source pulls from.")


class MakePredictionRequest(DatasetMakeRequest, PredictionParams):
    """Long-path request for kicking off a prediction: combines dataset construction + model parameters."""

    meta_data: dict = Field(
        default={}, description="Free-form metadata bag stored alongside the resulting prediction row."
    )


class MakeBacktestRequest(BacktestParams):
    """Request to backtest an already-imported dataset against a configured model."""

    name: str = Field(description="Human-friendly name for the resulting backtest row.")
    model_id: str = Field(description="Canonical name of the configured model to backtest.")
    dataset_id: int = Field(description="Foreign key to the dataset the backtest evaluates against.")


class MakeBacktestWithDataRequest(DatasetMakeRequest, BacktestParams):
    """Long-path request: build the dataset, then immediately backtest the configured model against it."""

    name: str = Field(description="Human-friendly name for the resulting backtest row.")
    model_id: str = Field(description="Canonical name of the configured model to backtest.")


class DataBaseResponse(DBModel):
    """Generic response that just echoes the created row's primary key."""

    id: int = Field(description="Primary key of the row that was created or affected.")


class DatasetCreate(DataSetCreateInfo):
    """Request body for creating a dataset directly from a fully-materialised observation list + polygons."""

    observations: list[ObservationBase] = Field(description="Every observation that should land in the new dataset.")
    geojson: FeatureCollectionModel = Field(description="GeoJSON polygon set for the dataset's org units.")


class ModelTemplateRead(DBModel, ModelTemplateInformation, ModelTemplateMetaData):
    """
    ModelTemplateRead is a read model for the ModelTemplateDB.
    It is used to return the model template in a readable format.
    """

    name: str = Field(description="Canonical unique identifier of the template.")
    id: int = Field(description="Primary key of the template.")
    user_options: dict | None = Field(
        default=None, description="JSON-schema-like dict describing the template's user-configurable options."
    )
    required_covariates: list[str] = Field(default=[], description="Covariate names the template must be given to run.")
    version: str | None = Field(default=None, description="Template version string, typically a git tag or commit sha.")
    archived: bool = Field(default=False, description="When True, the template is hidden from default pickers.")
    health_status: str | None = Field(
        default=None, description="Reported health status of the template, used by chapkit-hosted models."
    )
    uses_chapkit: bool = Field(
        default=False, description="When True, the template is served by a chapkit REST endpoint."
    )


class ConfiguredModelInfoRead(DBModel):
    """Detailed read view for a single configured model.

    Exposes the stored configuration (user option values, additional
    covariates) alongside the parent model template, so the frontend can
    render the user-option schema (e.g. the ``n_lags`` dynamic list) next
    to the chosen values without stitching together multiple list calls.
    """

    id: int = Field(description="Primary key of the configured model.")
    name: str = Field(description="Canonical name of the configured model.")
    display_name: str = Field(
        description="Human-friendly name stitched from the template name and (optionally) a configuration stub."
    )
    model_template_id: int = Field(description="Foreign key to the parent `ModelTemplateDB`.")
    user_option_values: dict | None = Field(
        default=None, description="Configured values for the template's user-options."
    )
    additional_continuous_covariates: list[str] = Field(
        default=[], description="Extra continuous covariates passed beyond the template's required set."
    )
    archived: bool = Field(default=False, description="When True, the configured model is hidden from default pickers.")
    uses_chapkit: bool = Field(
        default=False, description="Inherited from the template; True for chapkit-hosted models."
    )
    model_template: ModelTemplateRead = Field(description="Parent template the configuration extends.")


class ModelConfigurationCreate(DBModel):
    """Request body for adding a new configured model on top of a template."""

    name: str = Field(
        description="Canonical name for the new configured model; conventionally `<template_name>:<config_stub>`."
    )
    model_template_id: int = Field(description="Foreign key to the parent `ModelTemplateDB`.")
    user_option_values: dict = Field(
        default_factory=dict, description="Values for the user-options declared by the parent template."
    )
    additional_continuous_covariates: list[str] = Field(
        default=[], description="Extra continuous covariates beyond the template's required set."
    )


class PredictionCreate(DBModel):
    """Request body for creating a prediction row directly (DB-level)."""

    dataset_id: int = Field(description="Foreign key to the dataset the prediction was run against.")
    estimator_id: str = Field(description="Canonical name of the configured model used to produce the prediction.")
    n_periods: int = Field(description="Number of periods the model was asked to forecast.")


class BacktestUpdate(DBModel):
    """Partial-update body for an existing backtest. Currently only the human-friendly name is mutable."""

    name: str | None = Field(default=None, description="New human-friendly name; `None` leaves it unchanged.")


class PredictionSetupCreate(DBModel):
    """Request body for creating a recurring prediction setup attached to a backtest."""

    backtest_id: int = Field(description="Foreign key to the parent `Backtest` the setup will run forward in time.")
    name: str = Field(description="Human-friendly name for the setup.")
    schedule_cron_expression: str | None = Field(
        default=None, description="Standard cron expression for when to run; `None` means manual-only."
    )
    schedule_enabled: bool = Field(
        default=False, description="When True, the scheduler executes the setup at every cron tick."
    )
    quantile_targets: list[QuantileTarget] = Field(
        default_factory=list,
        description="Where to push each quantile of the predictive distribution.",
    )


class PredictionSetupUpdate(DBModel):
    """Partial-update body for an existing prediction setup. Rejects unknown fields with HTTP 422."""

    # Reject unknown fields so clients trying to update immutable fields
    # (backtestId, configuredModelId, etc.) get a clear 422 instead of a silent no-op.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")  # type: ignore[assignment]

    name: str | None = Field(default=None, description="New human-friendly name; `None` leaves it unchanged.")
    schedule_cron_expression: str | None = Field(
        default=None, description="New cron expression; `None` leaves it unchanged."
    )
    schedule_enabled: bool | None = Field(default=None, description="New enabled flag; `None` leaves it unchanged.")
    quantile_targets: list[QuantileTarget] | None = Field(
        default=None, description="New full quantile-targets list; `None` leaves it unchanged."
    )


class RunPredictionSetupRequest(DBModel):
    """Request body for a one-shot run of a prediction setup. Rejects unknown fields with HTTP 422."""

    # Reject unknown fields so legacy clients still sending dataSources /
    # dataToBeFetched / configuredModelWithDataSourceId fail loud instead of
    # having those fields silently dropped.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")  # type: ignore[assignment]

    name: str = Field(description="Human-friendly name for the one-shot prediction run.")
    geojson: FeatureCollectionModel = Field(description="GeoJSON polygon set for the org units in the run.")
    provided_data: list[ObservationBase] = Field(description="Observations supplied directly by the caller.")
    type: str | None = Field(default=None, description="Free-form run-type marker stored alongside the prediction.")
    n_periods: int = Field(default=3, gt=0, description="Number of future periods to forecast.")
