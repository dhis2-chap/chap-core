"""Data-access operations for datasets and their observations.

Extracted from ``SessionWrapper`` so dataset reads/writes live in one place.
``SessionWrapper`` keeps thin delegating methods for backwards compatibility.
"""

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import cast

from sqlmodel import select

from chap_core.datatypes import FullData, create_tsdataclass
from chap_core.geometry import Polygons
from chap_core.time_period import Month, Week

from ..spatio_temporal_data.converters import observations_to_dataset
from ..spatio_temporal_data.temporal_dataclass import DataSet as _DataSet
from .dataset_tables import DataSet, DataSetCreateInfo, DataSetInfo, Observation
from .manager import DbManager

logger = logging.getLogger(__name__)


def _filtered_observation_query(
    dataset_id: int,
    *,
    periods: list[str] | None = None,
    period_range: tuple[str, str] | None = None,
    org_units: list[str] | None = None,
    feature_names: list[str] | None = None,
):
    """Build a ``select(Observation)`` query for one dataset with optional filters."""
    expr = select(Observation).where(Observation.dataset_id == dataset_id)
    if periods is not None:
        expr = expr.where(Observation.period.in_(periods))  # type: ignore[attr-defined]
    if period_range is not None:
        start, end = period_range
        expr = expr.where(Observation.period >= start).where(Observation.period <= end)
    if org_units is not None:
        expr = expr.where(Observation.org_unit.in_(org_units))  # type: ignore[attr-defined]
    if feature_names is not None:
        expr = expr.where(Observation.feature_name.in_(feature_names))  # type: ignore[union-attr]
    return expr


class DataSetManager(DbManager[DataSet]):
    """Dataset data-access operations, backed by a SQLModel ``Session``."""

    model = DataSet

    def add_dataset_from_csv(self, name: str, csv_path: Path, geojson_path: Path | None = None):
        dataset = _DataSet.from_csv(csv_path, dataclass=FullData)
        geojson_content = open(geojson_path).read() if geojson_path else None
        features = None
        if geojson_content is not None:
            features = Polygons.from_geojson(json.loads(geojson_content), id_property="NAME_1").feature_collection()
            features = features.model_dump_json()

        return self.add_dataset(DataSetCreateInfo(name=name), dataset, features)

    def add_dataset(self, dataset_info: DataSetCreateInfo, orig_dataset: _DataSet, polygons):
        """
        Add a dataset to the database. The dataset is provided as a spatio-temporal dataclass.
        The polygons should be provided as a geojson feature collection.
        The dataset_info should contain information about the dataset, such as its name and data sources.
        The function sets some derived fields in the dataset_info, such as the first and last time period and the covariates.
        The function returns the id of the newly created dataset.
        """
        logger.info(
            f"Adding dataset {dataset_info.name} with {len(list(orig_dataset.locations()))} locations and {len(orig_dataset.period_range)} time periods"
        )
        field_names = [
            field.name
            for field in dataclasses.fields(next(iter(orig_dataset.values())))
            if field.name not in ["time_period", "location"]
        ]
        logger.info(f"Field names in dataset: {field_names}")
        if isinstance(orig_dataset.period_range[0], Month):
            period_type = "month"
        else:
            assert isinstance(orig_dataset.period_range[0], Week), orig_dataset.period_range[0]
            period_type = "week"
        full_info = DataSetInfo(
            first_period=orig_dataset.period_range[0].id,
            last_period=orig_dataset.period_range[-1].id,
            covariates=field_names,
            created=datetime.datetime.now(),
            org_units=list(orig_dataset.locations()),
            period_type=period_type,
            **dataset_info.model_dump(),
        )
        dataset = DataSet(geojson=polygons, **full_info.model_dump())

        for location, data in orig_dataset.items():
            field_names = [
                field.name for field in dataclasses.fields(data) if field.name not in ["time_period", "location"]
            ]
            for row in data:
                for field in field_names:
                    observation = Observation(
                        period=row.time_period.id,
                        org_unit=location,
                        value=float(getattr(row, field)),
                        feature_name=field,
                    )
                    dataset.observations.append(observation)

        self.add(dataset)
        assert self.session.exec(select(Observation).where(Observation.dataset_id == dataset.id)).first() is not None
        return dataset.id

    def get_observations(
        self,
        dataset_id: int,
        *,
        periods: list[str] | None = None,
        period_range: tuple[str, str] | None = None,
        org_units: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> list[Observation]:
        """Read raw observations for a dataset, optionally filtered.

        Unlike :meth:`get_dataset` this does not build a ``_DataSet`` domain object,
        so the selected periods do not need to be consecutive (an explicit, possibly
        non-contiguous ``periods`` list is allowed). Returns an empty list if the
        dataset does not exist or nothing matches the filters.
        """
        expr = _filtered_observation_query(
            dataset_id,
            periods=periods,
            period_range=period_range,
            org_units=org_units,
            feature_names=feature_names,
        )
        return list(self.session.exec(expr).all())

    def get_dataset(
        self,
        dataset_id: int,
        dataclass: type | None = None,
        *,
        period_range: tuple[str, str] | None = None,
        org_units: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> _DataSet:
        """Load a dataset as a ``_DataSet`` domain object, optionally filtered.

        The filters are restricted to ones that keep each location's series
        consecutive (``period_range``, ``org_units``, ``feature_names``); use
        :meth:`get_observations` for arbitrary period subsets. When ``feature_names``
        is given and ``dataclass`` is inferred, the inferred dataclass is narrowed to
        the requested features.
        """
        dataset = self.get(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset with id {dataset_id} not found")
        if dataclass is None:
            logger.info(f"Getting dataset with covariates: {dataset.covariates} and name: {dataset.name}")
            field_names = dataset.covariates
            if feature_names is not None:
                requested = set(feature_names)
                field_names = [name for name in field_names if name in requested]
            dataclass = create_tsdataclass(field_names)
        expr = _filtered_observation_query(
            dataset_id,
            period_range=period_range,
            org_units=org_units,
            feature_names=feature_names,
        )
        observations = list(self.session.exec(expr).all())
        if not observations:
            raise ValueError(f"No observations found for dataset {dataset_id} matching the given filters")
        new_dataset = observations_to_dataset(dataclass, observations)

        if dataset.geojson:
            logger.info(f"Loading polygons from geojson for dataset id {dataset_id}")
            new_dataset.set_polygons(Polygons.from_geojson(json.loads(dataset.geojson), id_property="district").data)

        return cast("_DataSet", new_dataset)

    def get_dataset_by_name(self, dataset_name: str) -> DataSet | None:
        dataset = self.session.exec(select(DataSet).where(DataSet.name == dataset_name)).first()
        return dataset
