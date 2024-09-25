import datetime
from pathlib import Path
from typing import Annotated, TypeAlias

import pytest
from chap_core.omnipy_lib import JsonScalar
# from omnipy.modules.json.typedefs import (
#   JsonScalar,
# )

from chap_core.spatio_temporal_data.omnipy_spatio_temporal_dataset import (
    SpatioTemporalDataOmnipyDataset,
    TemporalDataOmnipyDataset,
    TemporalDataPydanticModel,
    MultiResolutionTemporalDataOmnipyModel,
)

JsonTestDataType: TypeAlias = dict[
    str, dict[str, dict[str, list[dict[str, JsonScalar]]]]
]


@pytest.fixture
def simple_test_data() -> Annotated[JsonTestDataType, pytest.fixture]:
    return dict(
        region_1=dict(
            disease=dict(
                weeks=[
                    dict(start_date="2024-01-01", count=41),
                    dict(start_date="2024-01-08", count=30),
                    dict(start_date="2024-01-15", count=23),
                    dict(start_date="2024-01-22", count=38),
                    dict(start_date="2024-01-29", count=24),
                    dict(start_date="2024-02-05", count=19),
                    dict(start_date="2024-02-12", count=31),
                    dict(start_date="2024-02-19", count=35),
                    dict(start_date="2024-02-26", count=32),
                ]
            ),
            weather=dict(
                months=[
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
                ]
            ),
        ),
        region_2=dict(
            disease=dict(
                days=[
                    dict(
                        start_date=f"2024-{(i // 31) + 1:02d}-{(i % 31) + 1:02d}",
                        count=(i % 5) + 1,
                    )
                    for i in range(60)
                ]
            ),
            weather=dict(
                months=[
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
                ]
            ),
        ),
    )


class DiseaseFeatures(TemporalDataPydanticModel):
    count: int


class ClimateFeatures(TemporalDataPydanticModel):
    rain: float
    category: str


# class MyTemporalSubDatasetsModel(TemporalSubDatasetsPydanticModel):
#     disease: MultiResolutionTemporalDataOmnipyModel[DiseaseFeatures] = (
#         MultiResolutionTemporalDataOmnipyModel[DiseaseFeatures]()
#     )
#     weather: MultiResolutionTemporalDataOmnipyModel[ClimateFeatures] = (
#         MultiResolutionTemporalDataOmnipyModel[ClimateFeatures]()
#     )


class MyTemporalDataOmnipyDataset(TemporalDataOmnipyDataset):
    def _set_standard_field_description(self) -> None:
        super()._set_standard_field_description()

        self.set_model(
            "disease", MultiResolutionTemporalDataOmnipyModel[DiseaseFeatures]
        )
        self.set_model(
            "weather", MultiResolutionTemporalDataOmnipyModel[ClimateFeatures]
        )


def test_spatio_temporal_dataset(
    tmp_path: Annotated[Path, pytest.fixture],
    simple_test_data: Annotated[JsonTestDataType, pytest.fixture],
):
    persist_path = str(tmp_path / "simple_test_data")

    init_dataset = SpatioTemporalDataOmnipyDataset[MyTemporalDataOmnipyDataset](
        simple_test_data
    )
    init_dataset.save(persist_path)

    loaded_dataset = SpatioTemporalDataOmnipyDataset[MyTemporalDataOmnipyDataset]()
    loaded_dataset.load(persist_path, by_file_suffix=True)

    for dataset in (init_dataset, loaded_dataset):
        assert len(dataset) == 2
        for region, spatio_temp_data in dataset.items():
            # Usage of inner data objects is currently inconsistent with the outer dataset (which uses dict syntax) due to
            # switching between Omnipy datasets and Pydantic models explicitly wrapped as Omnipy models. Will be harmonised
            # in a newer version of Omnipy

            assert len(spatio_temp_data) == 2
            # assert hasattr(spatio_temp_data, "disease")
            # assert hasattr(spatio_temp_data, "weather")

            for category, multi_res_temp_data in spatio_temp_data.items():
                assert hasattr(multi_res_temp_data, "days")
                assert hasattr(multi_res_temp_data, "weeks")
                assert hasattr(multi_res_temp_data, "months")
                assert hasattr(multi_res_temp_data, "inconsistent")

                for resolution, temp_data in multi_res_temp_data:
                    for record in temp_data:
                        assert isinstance(record.start_date, datetime.date)
