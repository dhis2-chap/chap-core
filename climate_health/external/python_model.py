import logging
import subprocess

from ..datatypes import ClimateHealthTimeSeries, HealthData, ClimateData
from ..dataset import SpatioTemporalDataSet

from climate_health.time_period import Month
import tempfile


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

    def get_predictions(self, train_data: SpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: SpatioTemporalDataSet[ClimateData]) -> SpatioTemporalDataSet[HealthData]:
        # call command, read output results
        pass


def run_command(command: str):
    """Runs a unix command using subprocess"""
    command = command.split()
    try:
        output = subprocess.check_output(command)
        logging.info(output)
    except subprocess.CalledProcessError as e:
        error = e.output.decode()
        logging.info(error)
        raise e

    return output

class ExternalPythonModel:
    def __init__(self, script: str, lead_time=Month, adaptors=None):
        self._script = script
        self._lead_time = lead_time
        self._adaptors = adaptors

    def get_predictions(self, train_data: SpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: SpatioTemporalDataSet[ClimateData]) -> SpatioTemporalDataSet[HealthData]:

        with tempfile.NamedTemporaryFile() as out_file:
            command = f"python {self._script} {train_data} {future_climate_data} {out_file.name}"
            output = run_command(command)


