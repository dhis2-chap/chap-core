import itertools
from datetime import datetime

from geojson_pydantic import (
    Feature,
)
from geojson_pydantic import (
    FeatureCollection as _FeatureCollection,
)
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.alias_generators import to_camel
from sqlalchemy import JSON, Column, TypeDecorator, false
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID
from chap_core.time_period.date_util_wrapper import Month, TimePeriod, Week


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


def period_kind(period: TimePeriod) -> str:
    """Name the period type a masked range must share with its dataset.

    Monday and Sunday weeks are different kinds: their ids (`2021W01`, `2021SunW01`) never match.
    """
    if isinstance(period, Month):
        return "month"
    if isinstance(period, Week):
        return "sunday week" if "SunW" in period.id else "week"
    return type(period).__name__.lower()


def parse_masked_period(value: str) -> TimePeriod:
    """Parse a month or week period id the same way observation periods are parsed on import."""
    try:
        period = TimePeriod.parse(value)
    except Exception as e:  # TimePeriod.parse raises several unrelated error types for bad input
        raise ValueError(f"'{value}' is not a month or week period id") from e
    if not isinstance(period, Month | Week):
        raise ValueError(f"'{value}' is a {period_kind(period)}; masked ranges take month or week periods only")
    return period


class MaskedRange(DBModel):
    """An inclusive range of periods in which `disease_cases` is hidden from models (CLIM-1300).

    Both bounds are normalised to canonical period ids on creation, so `2021-6` and `202106` are stored
    as `202106`, and `2021W1` as `2021W01`.
    """

    start: PeriodID = Field(description="First masked period, a month or week id (e.g. `202106` or `2021W01`).")
    end: PeriodID = Field(description="Last masked period, inclusive, of the same period type as `start`.")
    name: str | None = Field(default=None, description="Optional label for the range, such as `covid`.")

    @model_validator(mode="after")
    def _normalise_bounds(self) -> "MaskedRange":
        start, end = parse_masked_period(self.start), parse_masked_period(self.end)
        if period_kind(start) != period_kind(end):
            raise ValueError(f"Range {self.start} to {self.end} mixes a {period_kind(start)} and a {period_kind(end)}")
        if start > end:
            raise ValueError(f"Range {self.start} to {self.end} ends before it starts")
        self.start, self.end = start.id, end.id
        return self


def check_masked_ranges(masked_ranges: list[MaskedRange] | None) -> list[MaskedRange] | None:
    """Reject ranges of different period types or that overlap each other."""
    if not masked_ranges:
        return masked_ranges
    parsed = sorted(
        ((parse_masked_period(r.start), parse_masked_period(r.end)) for r in masked_ranges),
        key=lambda bounds: bounds[0].start_timestamp,
    )
    if len({period_kind(start) for start, _ in parsed}) > 1:
        raise ValueError("Masked ranges must all use the same period type")
    for (_, previous_end), (start, _) in itertools.pairwise(parsed):
        if start <= previous_end:
            raise ValueError(f"Masked ranges overlap at {start.id}")
    return masked_ranges


def check_masked_ranges_match_dataset(masked_ranges: list[MaskedRange] | None, first_period: TimePeriod) -> None:
    """Raise if the ranges use a different period type than the dataset (weeks on a monthly dataset)."""
    for masked_range in masked_ranges or []:
        kind = period_kind(parse_masked_period(masked_range.start))
        if kind != period_kind(first_period):
            raise ValueError(
                f"Masked range {masked_range.start} to {masked_range.end} is a {kind} range, "
                f"but the dataset has {period_kind(first_period)} periods"
            )


class DataConfiguration(DBModel):
    """Data settings read from the CLI `--data-configuration` file, in the same shape as the REST dataset payload."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")  # type: ignore[assignment]

    masked_ranges: list[MaskedRange] = Field(
        default_factory=list, description="Ranges in which `disease_cases` is masked for every run on the data."
    )

    @field_validator("masked_ranges")
    @classmethod
    def _check_masked_ranges(cls, value: list[MaskedRange] | None) -> list[MaskedRange] | None:
        return check_masked_ranges(value)


class DataSetCreateInfoBase(DBModel):
    """Dataset metadata shared by every request that creates a dataset, including the inline-data ones."""

    name: str = Field(description="Name of dataset")

    data_sources: list[DataSource] | None = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(DataSource)),
        description="A mapping of covariate names to data element IDs from which to source the data",
    )
    type: str | None = Field(None, description="Purpose of dataset, e.g., 'forecasting' or 'backtesting'")


class DataSetCreateInfo(DataSetCreateInfoBase):
    """Metadata fields required when registering a new dataset.

    Masked ranges are set here, when a dataset is created, and never changed afterwards.
    """

    masked_ranges: list[MaskedRange] | None = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(MaskedRange)),
        description=(
            "Inclusive period ranges in which `disease_cases` is masked. Set at creation and never changed. "
            "Deletes data: `disease_cases` inside the ranges is stored as empty and the submitted values "
            "cannot be recovered."
        ),
    )

    @field_validator("masked_ranges")
    @classmethod
    def _check_masked_ranges(cls, value: list[MaskedRange] | None) -> list[MaskedRange] | None:
        return check_masked_ranges(value)


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
    created_manually: bool = Field(
        default=False,
        sa_column_kwargs={"server_default": false()},
        description="True if a user created the dataset directly, false if it was created as part of an evaluation or prediction.",
    )


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
