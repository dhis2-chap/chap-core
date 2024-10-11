from collections import defaultdict

from pydantic import BaseModel

from chap_core.datatypes import TimeSeriesArray


class DataElement(BaseModel):
    orgUnit: str
    period: str
    value: float


class SeasonalForecast:
    def __init__(self, data_dict: dict[str, dict[str, dict[str, float]]] | None = None):
        if data_dict is None:
            data_dict = {}
        self.data_dict = data_dict

    def add_json(self, field_name: str, json_data: list[DataElement]):
        data_dict = self.data_dict.get(field_name, defaultdict(dict))  # type: ignore
        orgUnits = []
        for data in json_data:
            data = DataElement(**data)  # type: ignore
            data_dict[data.orgUnit][data.period] = data.value
            orgUnits.append(data.orgUnit)
        print(f"Added periods {orgUnits} to field {field_name}")
        self.data_dict[field_name] = data_dict

    def get_forecasts(self, org_unit, period_range, field_name, start_date=None):
        assert field_name in self.data_dict, f"Field {field_name} not found in data {self.data_dict.keys()}"

        data = self.data_dict[field_name][org_unit]

        assert all(
            period.id in data for period in period_range
        ), f"Not all periods found in data {data.keys(), org_unit}"
        return TimeSeriesArray(period_range, [data[period.id] for period in period_range])
