from .external_model import run_command
from ..datatypes import ClimateHealthTimeSeries, HealthData, ClimateData
from .._legacy_dataset import IsSpatioTemporalDataSet

from chap_core.time_period import Month
import tempfile

from ..spatio_temporal_data.temporal_dataclass import DataSet


class ExternalPythonModel:
    def __init__(self, script: str, lead_time=Month, adaptors=None):
        self._script = script
        self._lead_time = lead_time
        self._adaptors = adaptors

    def train(self):
        pass

    def get_predictions(
        self,
        train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
        future_climate_data: IsSpatioTemporalDataSet[ClimateData],
    ) -> IsSpatioTemporalDataSet[HealthData]:
        train_data_file = tempfile.NamedTemporaryFile()
        future_climate_data_file = tempfile.NamedTemporaryFile()
        output_file = tempfile.NamedTemporaryFile()
        train_data.to_csv(train_data_file.name)
        future_climate_data.to_csv(future_climate_data_file.name)

        command = f"python {self._script} {train_data_file.name} " f"{future_climate_data_file.name} {output_file.name}"
        run_command(command)
        results = DataSet.from_csv(output_file.name, HealthData)
        train_data_file.close()
        future_climate_data_file.close()
        output_file.close()
        return results
