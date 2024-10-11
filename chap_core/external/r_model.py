from pathlib import Path
from typing import TypeVar, Generic
from chap_core.assessment.dataset_splitting import IsTimeDelta
from chap_core._legacy_dataset import IsSpatioTemporalDataSet
from chap_core.datatypes import ClimateHealthTimeSeries, HealthData, ClimateData
from chap_core.time_period import Month


class ExternalRModel:
    def __init__(self, r_script: str, lead_time=Month, adaptors=None):
        self.r_script = r_script
        self.lead_time = lead_time
        self.adaptors = adaptors

    def get_predictions(self, train_data: ClimateHealthTimeSeries, future_climate_data: ClimateData) -> HealthData:
        pass


FeatureType = TypeVar("FeatureType")


class ExternalLaggedRModel(Generic[FeatureType]):
    def __init__(
        self,
        script_file_name: str,
        data_type: type[FeatureType],
        tmp_dir: Path,
        lag_period: IsTimeDelta,
    ):
        self._script_file_name = script_file_name
        self._data_type = data_type
        self._tmp_dir = tmp_dir
        self._lag_period = lag_period
        self._saved_state = None
        self._model_filename = self._tmp_dir / "model.rds"

    def train(self, train_data: IsSpatioTemporalDataSet[FeatureType]):
        training_data_file = self._tmp_dir / "training_data.csv"
        train_data.to_csv(training_data_file)
        end_timestamp = train_data.end_timestamp
        self._saved_state = train_data.restrict_time_period(end_timestamp - self._lag_period, None)
        self._run_train_script(self._script_file_name, training_data_file, self._model_filename)

    def predict(self, future_data: IsSpatioTemporalDataSet[FeatureType]) -> IsSpatioTemporalDataSet[FeatureType]:
        full_data = self._join_state_and_future(future_data)
        full_data_path = self._tmp_dir / "full_data.csv"
        full_data.to_csv(full_data_path)
        output_file = self._tmp_dir / "output.csv"
        self._run_predict_script(self._script_file_name, self._model_filename, full_data_path, output_file)
        return self._read_output(output_file)
