import logging
from pathlib import Path

import pandas as pd

from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import HealthData, Samples
from chap_core.exceptions import CommandLineException, InvalidModelException, ModelFailedException, NoPredictionsError
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.geometry import Polygons
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import Month, TimePeriod

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
        if self._location_mapping is not None:  # type: ignore[attr-defined]
            data["location"] = data["location"].apply(self._location_mapping.name_to_index)  # type: ignore[attr-defined]
        if self._adapters is None:  # type: ignore[attr-defined]
            return data
        adapters = self._adapters  # type: ignore[attr-defined]
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
        name: str | None = None,
        adapters=None,
        working_dir=None,
        data_type=HealthData,
        configuration: ModelConfiguration | None = None,
        model_information: ModelTemplateConfigV2 | None = None,
        dry_run=False,
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
        self._polygons_file_name: Path | None = None
        self._configuration: dict[str, object] = (
            configuration or {}  # type: ignore[assignment]
        )  # configuration passed from the user to the model, e.g. about covariates or parameters
        # self._config_filename = "model_config.yaml"
        self._model_information = model_information
        self._dry_run = dry_run

    @property
    def name(self):
        return self._name

    @property
    def configuration(self):
        return self._configuration

    @property
    def model_information(self):
        return self._model_information

    def _apply_generated_features(self, dataset: DataSet) -> DataSet:
        """Apply any gen:-prefixed feature generators to the dataset."""
        from chap_core.feature_generators import apply_feature_generators, parse_generated_covariates

        if self._model_information is None:
            return dataset
        _, generator_ids = parse_generated_covariates(self._model_information.required_covariates)
        if generator_ids:
            dataset = apply_feature_generators(dataset, generator_ids)
        return dataset

    def _copy_generated_features(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        """Copy location-constant generated features from historic_data to future_data."""
        from chap_core.feature_generators import parse_generated_covariates

        if self._model_information is None:
            return future_data
        _, generator_ids = parse_generated_covariates(self._model_information.required_covariates)
        if not generator_ids:
            return future_data

        historic_df = historic_data.to_pandas()
        future_df = future_data.to_pandas()
        # For each generated feature, extract per-location values from historic data
        generated_fields = set(historic_df.columns) - set(future_df.columns)
        if not generated_fields:
            return future_data
        for field in generated_fields:
            loc_values = historic_df.groupby("location")[field].first()
            future_df[field] = future_df["location"].map(loc_values)
        return DataSet.from_pandas(future_df)

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

        train_data = self._apply_generated_features(train_data)

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

        # removed line below, writing this config is handled by runner, this should not be needed here
        # yaml.dump(self._configuration, open(self._config_filename, "w"))
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

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet | None:  # type: ignore[override]
        logging.debug("Running predict")
        historic_data = self._apply_generated_features(historic_data)
        future_data = self._copy_generated_features(historic_data, future_data)

        start_time = future_data.start_timestamp
        suffix = start_time.date.date().isoformat()
        future_data_name = Path(self._working_dir) / f"future_data_{suffix}.csv"
        historic_data_name = Path(self._working_dir) / f"historic_data_{suffix}.csv"
        logger.debug(f"Predicting on dataset from {start_time} to {future_data.end_timestamp}")

        for filename, dataset in [
            (future_data_name, future_data),
            (historic_data_name, historic_data),
        ]:
            with open(filename, "w"):
                adapted_dataset = self._adapt_data(dataset.to_pandas(), frequency=self._get_frequency(dataset))
                adapted_dataset.to_csv(filename)

        predictions_file = Path(self._working_dir) / f"predictions_{suffix}.csv"

        # touch predictions.csv
        with open(predictions_file, "w") as _:
            pass

        try:
            self._runner.predict(
                self._model_file_name,
                f"historic_data_{suffix}.csv",
                f"future_data_{suffix}.csv",
                f"predictions_{suffix}.csv",
                "polygons.geojson" if self._polygons_file_name is not None else None,
            )
        except CommandLineException as e:
            logger.error("Error predicting model, command failed")
            raise ModelFailedException(str(e)) from e

        if self._dry_run:
            return None

        try:
            df = pd.read_csv(predictions_file)
        except pd.errors.EmptyDataError:
            # todo: Probably deal with this in an other way, throw an exception istead
            logger.warning("No data returned from model (empty file from predictions)")
            raise NoPredictionsError("No prediction data written") from None

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
            raise ModelFailedException(f"Error while parsing predictions: {e}") from e

        return d

    def report(
        self,
        historic_data: DataSet,
        out_file: Path,
        model_artifact: Path | None = None,
    ) -> None:
        """Generate a PDF report using the model's report entry point.

        If ``model_artifact`` is provided, that path is passed as the ``{model}``
        parameter. Otherwise the in-memory model file from a prior ``train()`` call is used.
        """
        if self._model_information is None or self._model_information.entry_points is None:
            raise InvalidModelException("Model has no entry points configured; cannot generate report")
        if self._model_information.entry_points.report is None:
            raise InvalidModelException(f"Model '{self._name}' does not define a 'report' entry point")

        historic_data = self._apply_generated_features(historic_data)

        historic_data_name = Path(self._working_dir) / "report_historic_data.csv"
        adapted = self._adapt_data(historic_data.to_pandas(), frequency=self._get_frequency(historic_data))
        adapted.to_csv(historic_data_name)

        model_path_arg = str(model_artifact) if model_artifact is not None else self._model_file_name

        try:
            self._runner.report(
                model_path_arg,
                "report_historic_data.csv",
                str(out_file),
                "polygons.geojson" if self._polygons_file_name is not None else None,
            )
        except CommandLineException as e:
            logger.error("Error generating report, command failed")
            raise ModelFailedException(str(e)) from e

        self._runner.teardown()
