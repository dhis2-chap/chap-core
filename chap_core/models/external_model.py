import logging
from pathlib import Path
import pandas as pd
import yaml

from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import HealthData, Samples
from chap_core.exceptions import CommandLineException, ModelFailedException, NoPredictionsError
from chap_core.geometry import Polygons
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import TimePeriod, Month

logger = logging.getLogger(__name__)


def _extract_week_number(period_str: str) -> int:
    """Extract week number from period string.

    Handles both old format (2020W01, 2020SunW01) and new format (2020-W01, 2020-S01).
    """
    # New format: YYYY-Wnn or YYYY-Snn
    if "-W" in period_str:
        return int(period_str.split("-W")[-1])
    if "-S" in period_str:
        return int(period_str.split("-S")[-1])
    # Old format: YYYYSunWnn or YYYYWnn
    if "SunW" in period_str:
        return int(period_str.split("SunW")[-1])
    if "W" in period_str:
        return int(period_str.split("W")[-1])
    raise ValueError(f"Cannot extract week number from: {period_str}")


def _extract_year(period_str: str) -> int:
    """Extract year from period string.

    Handles both old format (2020W01) and new format (2020-W01, 2020-S01).
    """
    # New format: YYYY-Wnn or YYYY-Snn (year is before the hyphen)
    if "-W" in period_str or "-S" in period_str:
        return int(period_str.split("-")[0])
    # Old format: YYYYSunWnn or YYYYWnn
    if "SunW" in period_str:
        return int(period_str.split("SunW")[0])
    if "W" in period_str:
        return int(period_str.split("W")[0])
    # Monthly format: YYYY-MM
    if "-" in period_str:
        return int(period_str.split("-")[0])
    raise ValueError(f"Cannot extract year from: {period_str}")


class ExternalModelBase(ConfiguredModel):
    """
    A base class for external models that provides some utility methods"""

    def _adapt_data(self, data: pd.DataFrame, inverse=False, frequency="ME"):
        if self._location_mapping is not None:
            data["location"] = data["location"].apply(self._location_mapping.name_to_index)
        if self._adapters is None:
            return data
        adapters = self._adapters
        logger.info(f"Adapting data with columns {data.columns.tolist()} using adapters {adapters}")
        if inverse:
            adapters = {v: k for k, v in adapters.items()}
            # data['disease_cases'] = data[adapters['disase_cases']]
            return data

        for to_name, from_name in adapters.items():
            # ignore if the column is not present
            if from_name == "disease_cases" and "disease_cases" not in data.columns:
                continue

            if from_name == "week":
                if frequency == "W":
                    logger.info("Converting time period to week number")
                    if hasattr(data["time_period"], "dt"):
                        new_val = data["time_period"].dt.week
                        data[to_name] = new_val
                    else:
                        data[to_name] = [_extract_week_number(str(p)) for p in data["time_period"]]

            elif from_name == "month":
                if frequency == "ME":
                    logger.info("Converting time period to month number")

                    if hasattr(data["time_period"], "dt"):
                        data[to_name] = data["time_period"].dt.month
                    else:
                        data[to_name] = [int(str(p).split("-")[-1]) for p in data["time_period"]]

            elif from_name == "year":
                logger.info("Converting time period to year")
                if hasattr(data["time_period"], "dt"):
                    data[to_name] = data["time_period"].dt.year
                else:
                    data[to_name] = [_extract_year(str(p)) for p in data["time_period"]]
            else:
                data[to_name] = data[from_name]
        logger.info(f"Adapted data to columns {data.columns.tolist()}")
        return data

    def _write_polygons_to_geojson(self, dataset: DataSet, out_file_name):
        if dataset.polygons is not None:
            logging.info(f"Writing polygons to {out_file_name}")
            Polygons(dataset.polygons).to_file(out_file_name)

    def _get_frequency(self, train_data):
        frequency = "ME" if isinstance(train_data.period_range[0], Month) else "W"
        return frequency

    def __call__(self):
        return self


class ExternalModel(ExternalModelBase):
    """
    An ExternalModel is a specififc implementation of a ConfiguredModel that represents
    a model that is "external" in the sense that it needs to be run through a runner (e.g. a DockerRunner).
    This class is typically used for external models developed outside of Chap, and gives such models
    an interface with methods like train and predict so that they are compatible with Chap.
    """

    def __init__(
        self,
        runner,
        name: str = None,
        adapters=None,
        working_dir=None,
        data_type=HealthData,
        configuration: ModelConfiguration | None = None,
    ):
        self._runner = runner  # MlFlowTrainPredictRunner(model_path)
        # self.model_path = model_path
        self._adapters = adapters
        if working_dir is None:
            from chap_core import get_temp_dir

            working_dir = str(get_temp_dir() / "models")
        self._working_dir = working_dir
        self._location_mapping = None
        self._model_file_name = "model"
        self._data_type = data_type
        self._name = name
        self._polygons_file_name = None
        self._configuration = (
            configuration or {}
        )  # configuration passed from the user to the model, e.g. about covariates or parameters
        self._config_filename = "model_config.yaml"

    @property
    def name(self):
        return self._name

    @property
    def configuration(self):
        return self._configuration

    @property
    def required_fields(self):
        return self._required_fields

    @property
    def optional_fields(self):
        return self._optional_fields

    def train(self, train_data: DataSet, extra_args=None):
        """
        Trains this model on the given dataset.

        Parameters
        ----------
        train_data : DataSet
            The data to train the model on
        extra_args : str
            Extra arguments to pass to the train command
        """
        if extra_args is None:
            extra_args = ""

        train_file_name = "training_data.csv"
        train_file_name_full = Path(self._working_dir) / Path(train_file_name)
        if train_data.polygons is not None:
            self._polygons_file_name = Path(self._working_dir) / "polygons.geojson"
            self._write_polygons_to_geojson(train_data, self._polygons_file_name)
            logging.info(f"Will pass polygons file {self._polygons_file_name} to train command and predict command")

        frequency = self._get_frequency(train_data)
        pd = train_data.to_pandas()
        new_pd = self._adapt_data(pd, frequency=frequency)
        new_pd.to_csv(train_file_name_full)

        yaml.dump(self._configuration, open(self._config_filename, "w"))
        try:
            self._runner.train(
                train_file_name,
                self._model_file_name,
                polygons_file_name="polygons.geojson" if self._polygons_file_name is not None else None,
            )
        except CommandLineException as e:
            logger.error("Error training model, command failed")
            raise ModelFailedException(str(e)) from e
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        logging.debug("Running predict")
        future_data_name = Path(self._working_dir) / "future_data.csv"
        historic_data_name = Path(self._working_dir) / "historic_data.csv"
        start_time = future_data.start_timestamp
        logger.debug(f"Predicting on dataset from {start_time} to {future_data.end_timestamp}")

        for filename, dataset in [
            (future_data_name, future_data),
            (historic_data_name, historic_data),
        ]:
            with open(filename, "w"):
                adapted_dataset = self._adapt_data(dataset.to_pandas(), frequency=self._get_frequency(dataset))
                adapted_dataset.to_csv(filename)

        predictions_file = Path(self._working_dir) / "predictions.csv"

        # touch predictions.csv
        with open(predictions_file, "w") as _:
            pass

        try:
            self._runner.predict(
                self._model_file_name,
                "historic_data.csv",
                "future_data.csv",
                "predictions.csv",
                "polygons.geojson" if self._polygons_file_name is not None else None,
            )
        except CommandLineException as e:
            logger.error("Error predicting model, command failed")
            raise ModelFailedException(str(e)) from e

        try:
            df = pd.read_csv(predictions_file)
        except pd.errors.EmptyDataError:
            # todo: Probably deal with this in an other way, throw an exception istead
            logger.warning("No data returned from model (empty file from predictions)")
            raise NoPredictionsError("No prediction data written")

        if self._location_mapping is not None:
            df["location"] = df["location"].apply(self._location_mapping.index_to_name)

        time_periods = [TimePeriod.parse(s) for s in df.time_period.astype(str)]
        mask = [start_time <= time_period.start_timestamp for time_period in time_periods]
        df = df[mask]

        self._runner.teardown()

        try:
            d = DataSet.from_pandas(df, Samples)
        except ValueError as e:
            logging.error(f"Error while parsing predictions: {df}")
            logging.error(f"Error message: {e}")
            raise ModelFailedException("Error while parsing predictions: %s" % e)

        return d
