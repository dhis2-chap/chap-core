from typing import Annotated, TypeAlias

import pytest
from omnipy.modules.json.typedefs import (
    JsonScalar,
)

from climate_health.spatio_temporal_data.omnipy_spatio_temporal_dataset import (
    SpatioTemporalDataOmnipyDataset,
    TemporalDataModel,
    TemporalDataOmnipyDataset,
)


class FeaturesDatafileA(TemporalDataModel):
    count: int


class FeaturesDatafileB(TemporalDataModel):
    intensity: float
    category: str


JsonTestDataType: TypeAlias = dict[str, dict[str, list[dict[str, JsonScalar]]]]


@pytest.fixture
def simple_test_data() -> Annotated[JsonTestDataType, pytest.fixture]:
    return dict(
        region_1=dict(
            a=[
                dict(start_date="2024-01-01", period_length=7, count=41),
                dict(start_date="2024-01-08", period_length=7, count=30),
                dict(start_date="2024-01-15", period_length=7, count=23),
                dict(start_date="2024-01-22", period_length=7, count=38),
                dict(start_date="2024-01-29", period_length=7, count=24),
                dict(start_date="2024-02-05", period_length=7, count=19),
                dict(start_date="2024-02-12", period_length=7, count=31),
                dict(start_date="2024-02-19", period_length=7, count=35),
            ],
            b=[
                dict(
                    start_date="2024-01-01",
                    period_length=31,
                    intensity=12.5,
                    category="low",
                ),
                dict(
                    start_date="2024-02-01",
                    period_length=29,
                    intensity=24.3,
                    category="high",
                ),
            ],
        ),
        region_2=dict(
            a=[
                dict(start_date="2023-12-31", period_length=14, count=83),
                dict(start_date="2024-01-14", period_length=14, count=75),
                dict(start_date="2024-01-28", period_length=14, count=69),
                dict(start_date="2024-02-11", period_length=14, count=81),
                dict(start_date="2024-02-25", period_length=14, count=75),
            ],
            b=[
                dict(
                    start_date="2024-01-01",
                    period_length=31,
                    intensity=18.3,
                    category="medium",
                ),
                dict(
                    start_date="2024-02-01",
                    period_length=29,
                    intensity=19.2,
                    category="medium",
                ),
            ],
        ),
    )


def test_spatio_temporal_dataset(
    simple_test_data: Annotated[JsonTestDataType, pytest.fixture]
):
    dataset = SpatioTemporalDataOmnipyDataset(simple_test_data)
    # dataset.set_model("a", TemporalDataOmnipyDataset[FeaturesDatafileA])
    # dataset.set_model("b", TemporalDataOmnipyDataset[FeaturesDatafileB])
    assert len(dataset) == 2
