import numpy as np
import pandas as pd

from chap_core.datatypes import HealthData, HealthPopulationData, Samples
from chap_core.dhis2_interface.periods import (
    get_period_id,
    convert_time_period_string,
)
from chap_core.dhis2_interface.src.PushResult import DataValue
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import logging

logger = logging.getLogger(__name__)


class MetadDataLookup:
    def __init__(self, meta_data_json):
        self._lookup = {name: value["name"] for name, value in meta_data_json["items"].items()}

    def __getitem__(self, item):
        return self._lookup[item]

    def __contains__(self, item):
        return item in self._lookup


def parse_population_data(json_data, field_name="GEN - Population", col_idx=1):
    logger.warning("Only using one population number per location")
    # meta_data = MetadDataLookup(json_data["metaData"])
    lookup = {}
    for row in json_data["rows"]:
        # if meta_data[row[0]] != field_name:
        #    continue
        lookup[row[col_idx]] = float(row[3])
    return lookup


def parse_climate_data(json_data):
    # PARSE DATA HERE
    return


def parse_disease_data(
    json_data,
    disease_name="IDS - Dengue Fever (Suspected cases)",
    name_mapping={"time_period": 1, "disease_cases": 3, "location": 2},
):
    # meta_data = MetadDataLookup(json_data['metaData'])
    df = json_to_pandas(json_data, name_mapping)
    return DataSet.from_pandas(df, dataclass=HealthData, fill_missing=True)


def parse_json_rows(rows, name_mapping):
    new_rows = []
    col_names = list(name_mapping.keys())
    for row in rows:
        new_row = row
        new_rows.append([new_row[name_mapping[col_name]] for col_name in col_names])
    return new_rows


def json_to_pandas(json_data, name_mapping):
    new_rows = []
    col_names = list(name_mapping.keys())
    # col_names = ['time_period', 'disease_cases', 'location']
    for row in json_data["rows"]:
        # if meta_data[row[0]] != disease_name:
        #    continue
        new_row = row
        # new_row[name_mapping['location']] = meta_data[new_row[name_mapping['location']]]
        # new_row = [meta_data[elem] if elem in meta_data else elem for elem in row]
        new_rows.append([new_row[name_mapping[col_name]] for col_name in col_names])
    df = pd.DataFrame(new_rows, columns=col_names)
    df["week_id"] = [get_period_id(row) for row in df["time_period"]]
    df["time_period"] = [convert_time_period_string(row) for row in df["time_period"]]
    df.sort_values(by=["location", "week_id"], inplace=True)
    return df


def join_data(json_data, population_data):
    population_lookup = parse_population_data(population_data, col_idx=2)
    disease_data = parse_disease_data(json_data)
    return add_population_data(disease_data, population_lookup)


def add_population_data(disease_data, population_lookup):
    new_dict = {
        location: HealthPopulationData(
            data.data().time_period,
            data.data().disease_cases,
            np.full(len(data.data()), population_lookup[location]),
        )
        for location, data in disease_data.items()
    }
    return DataSet(new_dict)


def samples_to_datavalue(data: DataSet[Samples], attribute_mapping):
    pass


def predictions_to_datavalue(data: DataSet[HealthData], attribute_mapping: dict[str, str]):
    entries = []
    for location, data in data.items():
        data = data.data()
        for i, time_period in enumerate(data.time_period):
            for from_name, to_name in attribute_mapping.items():
                entry = DataValue(
                    getattr(data, from_name)[i],
                    location,
                    to_name,
                    time_period.to_string().replace("-", ""),
                )

                entries.append(entry)
    return entries
