from collections import defaultdict

from pydantic import BaseModel

from climate_health.datatypes import TimeSeriesArray
from climate_health.time_period import PeriodRange


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
        for data in json_data:
            data = DataElement(**data)
            data_dict[data.orgUnit][data.period] = data.value

        self.data_dict[field_name] = data_dict

    def get_forecasts(self, org_unit, period_range, field_name, start_date=None):
        data = self.data_dict[field_name][org_unit]
        return TimeSeriesArray(period_range, [data[period.id] for period in period_range])
