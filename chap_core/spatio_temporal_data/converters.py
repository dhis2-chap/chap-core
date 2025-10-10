import pandas as pd

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.dataset_tables import DataSet
from chap_core.datatypes import create_tsdataclass
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as _DataSet


def observations_to_dataset(dataclass, observations, fill_missing=False):
    obs_dicts = [obs.model_dump() for obs in observations]
    dataframe = pd.DataFrame(obs_dicts).rename(columns={"org_unit": "location", "period": "time_period"})
    dataframe = dataframe.set_index(["location", "time_period"])
    pivoted = dataframe.pivot(columns="feature_name", values="value").reset_index()
    new_dataset = _DataSet.from_pandas(pivoted, dataclass, fill_missing=fill_missing)
    return new_dataset


def dataset_model_to_dataset(dataset: DataSet):
    dataclass = create_tsdataclass(dataset.covariates)
    ds = observations_to_dataset(dataclass, dataset.observations)
    if dataset.geojson is not None:
        polygons = FeatureCollectionModel.model_validate(dataset.geojson)
        ds.set_polygons(polygons)
    return ds
