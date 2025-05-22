import pandas as pd

from chap_core.api_types import FeatureCollectionModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as _DataSet


def observations_to_dataset(dataclass, observations, fill_missing=False):
    dataframe = pd.DataFrame([obs.model_dump() for obs in observations]).rename(
        columns={"org_unit": "location", "period": "time_period"}
    )
    dataframe = dataframe.set_index(["location", "time_period"])
    pivoted = dataframe.pivot(columns="feature_name", values="value").reset_index()
    new_dataset = _DataSet.from_pandas(pivoted, dataclass, fill_missing=fill_missing)
    return new_dataset


def dataset_model_to_dataset(dataclass, dataset, fill_missing=False):
    ds = observations_to_dataset(dataclass, dataset.observations, fill_missing=fill_missing)
    polygons = FeatureCollectionModel.model_validate(dataset.geojson)
    ds.set_polygons(polygons)
    return ds
