import pytest
from geojson_pydantic import Point

from chap_core.api_types import (
    DataElement,
    DataList,
    FeatureCollectionModel,
    FeatureModel,
    PredictionRequest,
    RequestV1,
)
from chap_core.rest_api.worker_functions import (
    get_combined_dataset,
    get_health_dataset,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def test_get_health_dataset():
    prediction_request = PredictionRequest(
        orgUnitsGeoJson=FeatureCollectionModel(
            type="FeatureCollection",
            features=[
                FeatureModel(
                    id="location1",
                    properties={"name": "location1"},
                    geometry=Point(type="Point", coordinates=[0, 0]),  # type: ignore[reportArgumentType]
                    type="Feature",
                )
            ],
        ),
        features=[
            DataList(
                dhis2Id="disease_cases",
                featureId="disease_cases",
                data=[DataElement(pe="202101", value=1, ou="location1")],
            ),
            DataList(
                dhis2Id="population", featureId="population", data=[DataElement(pe="202101", value=1, ou="location1")]
            ),
            DataList(
                dhis2Id="rainfall", featureId="rainfall", data=[DataElement(pe="202101", value=1, ou="location1")]
            ),
            DataList(
                dhis2Id="something",
                featureId="mean_temperature",
                data=[DataElement(pe="202101", value=23.0, ou="location1")],
            ),
        ],
        include_data=True,
    )
    data = get_health_dataset(prediction_request)
    assert data["location1"].disease_cases[0] == 1
    assert data["location1"].mean_temperature[0] == 23.0


def test_dataset_from_request_v1(big_request_json):
    from chap_core.rest_api.worker_functions import dataset_from_request_v1

    data = PredictionRequest.model_validate_json(big_request_json)
    dataset = dataset_from_request_v1(data)
    print(dataset)


@pytest.fixture
def combined_dataset():
    periods = ["202101", "202102"]
    locations = ["location1", "location2"]
    data_elements = ["disease_cases", "population", "rainfall"]
    data_lists = [
        DataList(
            dhis2Id="wierd" + data_element,
            featureId=data_element,
            data=[
                DataElement(pe=period, value=i * j * k % 7, ou=location)
                for i, period in enumerate(periods)
                for j, location in enumerate(locations)
            ],
        )
        for k, data_element in enumerate(data_elements)
    ]
    geojson = FeatureCollectionModel(
        type="FeatureCollection",
        features=[
            FeatureModel(
                id=location,
                properties={"name": location},
                geometry=Point(type="Point", coordinates=[0, 0]),  # type: ignore[reportArgumentType]
                type="Feature",
            )
            for location in locations
        ],
    )

    return RequestV1(orgUnitsGeoJson=geojson, features=data_lists)


def test_get_combined_dataset(combined_dataset):
    dataset = get_combined_dataset(combined_dataset)
    assert isinstance(dataset, DataSet)
    for location, data in dataset.items():
        assert len(data) == 2
        assert len(data.rainfall) == 2
        assert len(data.population) == 2
