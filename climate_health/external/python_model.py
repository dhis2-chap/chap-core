from .external_model import run_command
from ..datatypes import ClimateHealthTimeSeries, HealthData, ClimateData
from ..dataset import IsSpatioTemporalDataSet

from climate_health.time_period import Month
import tempfile

from ..spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class ExternalCommandLineModel:
    """
    Represents a model that can be run on the command line
    Optionally takes a path to a yaml file that defines a conda environment that
    will be created and used when running the model.
    """

    def __init__(self, command_template: str, conda_environment_yaml: str = None, lead_time=Month, adaptors=None):
        self.command_template = command_template
        self.lead_time = lead_time
        self.adaptors = adaptors

    def _get_command(self, train_data, future_climate_):
        return self.command_template.format(train_data=train_data, future_climate=future_climate_)

    def get_predictions(self, train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: IsSpatioTemporalDataSet[ClimateData]) -> IsSpatioTemporalDataSet[HealthData]:
        # call command, read output results
        pass


class ExternalPythonModel:
    def __init__(self, script: str, lead_time=Month, adaptors=None):
        self._script = script
        self._lead_time = lead_time
        self._adaptors = adaptors

    def get_predictions(self, train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: IsSpatioTemporalDataSet[ClimateData]) -> IsSpatioTemporalDataSet[HealthData]:

        train_data_file = tempfile.NamedTemporaryFile()
        future_climate_data_file = tempfile.NamedTemporaryFile()
        output_file = tempfile.NamedTemporaryFile()
        train_data.to_csv(train_data_file.name)
        future_climate_data.to_csv(future_climate_data_file.name)

        command = (f"python {self._script} {train_data_file.name} "
                    f"{future_climate_data_file.name} {output_file.name}")
        output = run_command(command)
        results = SpatioTemporalDict.from_csv(output_file.name, HealthData)

        train_data_file.close()
        future_climate_data_file.close()
        output_file.close()
        return results

