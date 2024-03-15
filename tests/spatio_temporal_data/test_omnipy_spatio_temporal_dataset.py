from typing import Annotated, TypeAlias

import pytest
from omnipy.modules.json.typedefs import (
    JsonScalar,
)
from pydantic import BaseModel

from climate_health.spatio_temporal_data.omnipy_spatio_temporal_dataset import (
    SpatioTemporalDataOmnipyDataset,
    TemporalDataModel,
    TemporalDataOmnipyModel,
)

JsonTestDataType: TypeAlias = dict[str, dict[str, list[dict[str, JsonScalar]]]]


@pytest.fixture
def simple_test_data() -> Annotated[JsonTestDataType, pytest.fixture]:
    return dict(
        region_1=dict(
            disease=[
                dict(start_date="2024-01-01", count=41),
                dict(start_date="2024-01-08", count=30),
                dict(start_date="2024-01-15", count=23),
                dict(start_date="2024-01-22", count=38),
                dict(start_date="2024-01-29", count=24),
                dict(start_date="2024-02-05", count=19),
                dict(start_date="2024-02-12", count=31),
                dict(start_date="2024-02-19", count=35),
                dict(start_date="2024-02-26", count=32),
            ],
            weather=[
                dict(
                    start_date="2024-01-01",
                    rain=12.5,
                    category="low",
                ),
                dict(
                    start_date="2024-02-01",
                    rain=24.3,
                    category="high",
                ),
            ],
        ),
        region_2=dict(
            disease=[
                dict(
                    start_date=f"2024-{(i // 31) + 1:02d}-{(i % 31) + 1:02d}",
                    count=(i % 5) + 1,
                )
                for i in range(60)
            ],
            weather=[
                dict(
                    start_date="2024-01-01",
                    rain=18.3,
                    category="medium",
                ),
                dict(
                    start_date="2024-02-01",
                    rain=19.2,
                    category="medium",
                ),
            ],
        ),
    )


class DiseaseFeatures(TemporalDataModel):
    count: int


class ClimateFeatures(TemporalDataModel):
    rain: float
    category: str


class MyTemporalDataModel(BaseModel):
    disease: TemporalDataOmnipyModel[DiseaseFeatures]
    weather: TemporalDataOmnipyModel[ClimateFeatures]


def test_spatio_temporal_dataset(
    simple_test_data: Annotated[JsonTestDataType, pytest.fixture]
):
    dataset = SpatioTemporalDataOmnipyDataset[MyTemporalDataModel](simple_test_data)
    assert len(dataset) == 2
