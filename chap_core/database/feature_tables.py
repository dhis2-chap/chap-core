from sqlalchemy import JSON, Column
from sqlmodel import Field

from chap_core.database.base_tables import DBModel
from chap_core.model_spec import PeriodType


class FeatureTypeBase(DBModel):
    """Shared display metadata for a covariate type (rainfall, temperature, ...).

    Used as a mixin by `FeatureType` (the DB-backed row keyed by canonical
    ``name``) and `FeatureTypeRead` (the API read shape).
    """

    display_name: str = Field(description="Human-friendly name shown to operators in pickers and plot titles.")
    description: str = Field(description="Short paragraph explaining what the feature represents.")


class FeatureTypeRead(FeatureTypeBase):
    """A covariate type as returned by the API. Same shape as the DB row."""

    name: str = Field(description="Canonical machine-readable identifier (e.g. `rainfall`, `mean_temperature`).")


class FeatureType(FeatureTypeBase, table=True):
    """Catalogue row for one covariate type that models can request as input."""

    name: str = Field(
        primary_key=True,
        description="Canonical machine-readable identifier (e.g. `rainfall`, `mean_temperature`).",
    )


class FeatureSource(DBModel, table=True):
    """A registered provider that can supply data for a specific feature type."""

    name: str = Field(primary_key=True, description="Canonical identifier of the feature source.")
    display_name: str = Field(description="Human-friendly name shown in source pickers.")
    feature_type: str = Field(
        foreign_key="featuretype.name",
        description="Canonical name of the `FeatureType` this source provides values for.",
    )
    provider: str = Field(description="Upstream provider this source pulls from (e.g. `era5`, `dhis2`).")
    supported_period_types: list[PeriodType] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Period granularities this source can deliver values at.",
    )


class ModelFeatureLink(DBModel, table=True):
    """Join row linking a `ModelSpec` to one of its supported feature types."""

    model_id: int | None = Field(
        default=None,
        foreign_key="modelspec.id",
        primary_key=True,
        description="Primary-key id of the linked `ModelSpec` row.",
    )
    feature_type: str | None = Field(
        default=None,
        foreign_key="featuretype.name",
        primary_key=True,
        description="Canonical name of the linked `FeatureType` row.",
    )
