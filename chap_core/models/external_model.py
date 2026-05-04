import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import HealthData, Samples
from chap_core.exceptions import CommandLineException, ModelFailedException, NoPredictionsError
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.geometry import Polygons
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import Month, TimePeriod

logger = logging.getLogger(__name__)


def _positive_int_env(name: str, default: int) -> int:
    """Parse a positive integer environment variable, raising a clear error on failure."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}={raw!r} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name}={raw!r} must be a positive integer")
    return value


def _numeric_columns(df: pd.DataFrame, cols: list[str], kind: str) -> np.ndarray:
    """Coerce the given dataframe columns to a float numpy matrix, attributing errors per-column."""
    out = []
    for c in cols:
        try:
            out.append(pd.to_numeric(df[c], errors="raise"))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"shap_values.csv column '{c}' ({kind}) contains non-numeric data: {exc}") from exc
    return np.column_stack([s.to_numpy(dtype=float, copy=False) for s in out])


def _parse_shap_csv(shap_file: Path) -> dict:
    """Parse shap_values.csv into a structured dict.

    Validates size and column counts (configurable via CHAP_NATIVE_SHAP_MAX_BYTES
    and CHAP_NATIVE_SHAP_MAX_FEATURES environment variables) before reading,
    rejects NaNs in both shap__ and value__ columns, and uses vectorised numpy
    access instead of iterrows.
    """
    max_bytes = _positive_int_env("CHAP_NATIVE_SHAP_MAX_BYTES", 50 * 1024 * 1024)
    max_features = _positive_int_env("CHAP_NATIVE_SHAP_MAX_FEATURES", 500)

    size = shap_file.stat().st_size
    if size > max_bytes:
        raise ValueError(f"shap_values.csv ({size} bytes) exceeds CHAP_NATIVE_SHAP_MAX_BYTES={max_bytes}")

    shap_df = pd.read_csv(shap_file)
    required = {"location", "time_period", "expected_value"}
    missing_required = sorted(required - set(shap_df.columns))
    if missing_required:
        raise ValueError(f"Missing required columns in shap_values.csv: {missing_required}")

    shap_prefixed = [c for c in shap_df.columns if c.startswith("shap__")]
    if not shap_prefixed:
        raise ValueError("shap_values.csv must contain at least one 'shap__<feature>' column")
    if len(shap_prefixed) > max_features:
        raise ValueError(
            f"shap_values.csv has too many SHAP feature columns ({len(shap_prefixed)} > "
            f"CHAP_NATIVE_SHAP_MAX_FEATURES={max_features})"
        )

    feature_names = [c[len("shap__") :] for c in shap_prefixed]
    missing_value_columns = [f"value__{f}" for f in feature_names if f"value__{f}" not in shap_df.columns]
    if missing_value_columns:
        raise ValueError(
            "shap_values.csv must include matching value columns for every SHAP feature. "
            f"Missing: {missing_value_columns}"
        )

    shap_cols = [f"shap__{f}" for f in feature_names]
    value_cols = [f"value__{f}" for f in feature_names]

    shap_matrix = _numeric_columns(shap_df, shap_cols, "shap")
    value_matrix = _numeric_columns(shap_df, value_cols, "value")

    if np.isnan(shap_matrix).any():
        bad = np.argwhere(np.isnan(shap_matrix))[0]
        raise ValueError(
            f"shap_values.csv contains NaN value in shap column '{shap_cols[int(bad[1])]}' at row={int(bad[0])}"
        )
    if np.isnan(value_matrix).any():
        bad = np.argwhere(np.isnan(value_matrix))[0]
        raise ValueError(
            f"shap_values.csv contains NaN feature value at row={int(bad[0])} column='{value_cols[int(bad[1])]}'"
        )

    locations = shap_df["location"].astype(str).to_numpy()
    periods = shap_df["time_period"].astype(str).to_numpy()
    expected_values = shap_df["expected_value"].astype(float).to_numpy()

    values_df = pd.DataFrame(
        {
            "location": locations,
            "time_period": periods,
            "expected_value": expected_values,
        }
    )
    values_df["shap_values"] = shap_matrix.tolist()
    feature_values_records = pd.DataFrame(value_matrix, columns=feature_names).to_dict("records")
    values_df["feature_values"] = pd.Series(feature_values_records, dtype=object)
    values = values_df.to_dict("records")

    global_expected = float(expected_values.mean()) if len(expected_values) else 0.0
    return {
        "feature_names": feature_names,
        "expected_value": global_expected,
        "values": values,
    }


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
        provides_native_shap: bool = False,
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
        self._provides_native_shap = provides_native_shap

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

        provides_native_shap = self._provides_native_shap or (
            self._model_information is not None and self._model_information.provides_native_shap
        )
        if provides_native_shap:
            shap_file = Path(self._working_dir) / "shap_values.csv"
            if shap_file.exists():
                try:
                    native_shap = _parse_shap_csv(shap_file)
                    d.native_shap = native_shap
                    logger.info("Loaded native SHAP values from shap_values.csv")
                except Exception as exc:
                    logger.warning("Failed to parse shap_values.csv: %s", exc)

        return d
