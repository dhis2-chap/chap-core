from datetime import datetime

from geojson_pydantic import (
    Feature,
)
from geojson_pydantic import (
    FeatureCollection as _FeatureCollection,
)
from sqlalchemy import JSON, Column, TypeDecorator
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID


class FeatureCollectionModel(_FeatureCollection):
    """GeoJSON `FeatureCollection` carrying the polygons for a dataset's org units."""

    features: list[Feature] = Field(description="One feature per org unit in the polygon set.")  # type: ignore[assignment]


class PydanticListType(TypeDecorator):
    """Custom SQLAlchemy type that automatically serializes/deserializes Pydantic model lists"""

    impl = JSON
    cache_ok = True

    def __init__(self, pydantic_model_class):
        self.pydantic_model_class = pydantic_model_class
        super().__init__()

    def process_bind_param(self, value, dialect):
        """Convert Python list of Pydantic models to JSON for database storage"""
        if value is None:
            return None
        if isinstance(value, list):
            return [item.model_dump() if hasattr(item, "model_dump") else item for item in value]
        return value

    def process_result_value(self, value, dialect):
        """Convert JSON from database back to list of Pydantic models"""
        if value is None:
            return []
        if isinstance(value, list):
            return [self.pydantic_model_class(**item) if isinstance(item, dict) else item for item in value]
        return value


class ObservationBase(DBModel):
    """A single observed value for one (period, org unit, feature) triple."""

    period: PeriodID = Field(
        description="Period the observation belongs to, encoded as an ISO-like string (e.g. `2024-W12`, `202403`)."
    )
    org_unit: str = Field(description="Identifier of the org unit (province, district, ...) the observation is from.")
    value: float | None = Field(description="Observed value. `None` is allowed for known-missing observations.")
    feature_name: str | None = Field(description="Canonical name of the `FeatureType` this observation is a value for.")


class Observation(ObservationBase, table=True):
    """Persisted observation row, owned by a parent `DataSet`."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    dataset_id: int = Field(foreign_key="dataset.id", description="Foreign key to the parent `DataSet`.")
    dataset: "DataSet" = Relationship(back_populates="observations")


class DataSource(DBModel):
    """Mapping from a covariate name to the DHIS2 data element id used to source it."""

    covariate: str = Field(description="Canonical covariate name (matching a `FeatureType.name`).")
    data_element_id: str = Field(description="External identifier of the data element to pull values from.")


class DataSetCreateInfo(DBModel):
    """Metadata fields required when registering a new dataset."""

    name: str = Field(description="Name of dataset")

    data_sources: list[DataSource] | None = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(DataSource)),
        description="A mapping of covariate names to data element IDs from which to source the data",
    )
    type: str | None = Field(None, description="Purpose of dataset, e.g., 'forecasting' or 'backtesting'")


class DataSetInfo(DataSetCreateInfo):
    """Summary view of a dataset — what was created plus what was derived after import."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    covariates: list["str"] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Names of the covariates (features) actually present in the dataset's observations.",
    )
    first_period: PeriodID | None = Field(default=None, description="Earliest period present in the observations.")
    last_period: PeriodID | None = Field(default=None, description="Latest period present in the observations.")
    org_units: list["str"] | None = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Identifiers of every org unit that has at least one observation.",
    )
    created: datetime | None = Field(default=None, description="Server-side timestamp when the dataset was registered.")

    period_type: str | None = Field(default=None, description="Granularity of the periods (`month`, `week`, ...).")


class DataSetBase(DataSetInfo):
    """`DataSetInfo` plus the polygon geojson stored alongside the dataset."""

    geojson: str | None = Field(
        default=None, description="GeoJSON `FeatureCollection` for the dataset's org units, stored as a string."
    )


class DataSet(DataSetBase, table=True):
    """Persisted dataset row with its child observations."""

    observations: list[Observation] = Relationship(back_populates="dataset", cascade_delete=True)


class DataSetWithObservations(DataSetBase):
    """Read view that bundles a dataset together with its observations."""

    id: int = Field(description="Primary key of the dataset.")
    observations: list[ObservationBase] = Field(description="Every observation belonging to this dataset.")
    created: datetime | None = Field(description="Server-side timestamp when the dataset was registered.")
