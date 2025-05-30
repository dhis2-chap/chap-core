import dataclasses
import datetime
import time
from typing import Optional, List, Iterable

import numpy as np
import psycopg2
import sqlalchemy
from chap_core.predictor.naive_estimator import NaiveEstimator
from sqlmodel import SQLModel, create_engine, Session, select
from .tables import BackTest, BackTestForecast, BackTestMetric, Prediction, PredictionSamplesEntry
from .model_spec_tables import ModelSpecRead
from .model_templates_and_config_tables import ModelTemplateDB, ConfiguredModelDB, ModelConfiguration
from .debug import DebugEntry
from .dataset_tables import Observation, DataSet
from chap_core.datatypes import create_tsdataclass

# CHeck if CHAP_DATABASE_URL is set in the environment
import os

from chap_core.time_period import TimePeriod
from .. import ModelTemplateInterface
from ..external.model_configuration import ModelTemplateConfigV2
from ..models import ModelTemplate
from ..models.configured_model import ConfiguredModel
from ..rest_api_src.data_models import BackTestCreate
from ..spatio_temporal_data.converters import observations_to_dataset
from ..spatio_temporal_data.temporal_dataclass import DataSet as _DataSet
import logging

logger = logging.getLogger(__name__)
engine = None
database_url = os.getenv("CHAP_DATABASE_URL", default=None)
logger.info(f"Database url: {database_url}")
if database_url is not None:
    n = 0
    while n < 30:
        try:
            engine = create_engine(database_url)
            break
        except sqlalchemy.exc.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}. Trying again")
            n += 1
            time.sleep(1)
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}. Trying again")
            n += 1
            time.sleep(1)
    else:
        raise ValueError("Failed to connect to database")
else:
    logger.warning("Database url not set. Database operations will not work")


# metric functions
# each function is run at each forecasted value
# each function must take samples_values, observation, and evaluation_results
# NOTE: evaluation_results is provided in order to provide greater context of all forecast samples and observed values
# TODO: passing evaluation_results is a bit hacky, not always needed, and not very clean (need a better approach)
# TODO: move below to separate metrics file
# TODO: probably also move to classes and more flexible


def crps_ensemble_timestep(sample_values: np.ndarray, obs: float, evaluation_results: Iterable[DataSet]) -> float:
    term1 = np.mean(np.abs(sample_values - obs))
    term2 = 0.5 * np.mean(np.abs(sample_values[:, None] - sample_values[None, :]))
    return float(term1 - term2)


def crps_ensemble_timestep_normalized(sample_values: np.ndarray, obs: float, evaluation_results: Iterable[DataSet]) -> float:
    crps = crps_ensemble_timestep(sample_values, obs, evaluation_results)
    obs_values = [
        cases
        for eval_result in evaluation_results
        for location, samples_with_truth in eval_result.items()
        for cases in samples_with_truth.disease_cases
    ]
    obs_min, obs_max = min(obs_values), max(obs_values)
    crps_norm = crps / (obs_max - obs_min)
    return float(crps_norm)


def _is_within_percentile(sample_values: np.ndarray, obs: float, lower_percentile: float, higher_percentile: float) -> float:
    low,high = np.percentile(sample_values, [lower_percentile, higher_percentile])
    is_within_range = 1 if (low <= obs <= high) else 0
    return float(is_within_range)


def is_within_10th_90th(sample_values: np.ndarray, obs: float, evaluation_results: Iterable[DataSet]) -> float:
    return _is_within_percentile(sample_values, obs, 10, 90)


def is_within_25th_75th(sample_values: np.ndarray, obs: float, evaluation_results: Iterable[DataSet]) -> float:
    return _is_within_percentile(sample_values, obs, 25, 75)


class SessionWrapper:
    """
    This is a wrapper around data access operations
    """

    def __init__(self, local_engine=None, session=None):
        self.engine = local_engine  #  or engine
        self.session: Optional[Session] = session

    def __enter__(self):
        self.session = Session(self.engine)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        return False

    def list_all(self, model):
        return self.session.exec(select(model)).all()

    def create_if_not_exists(self, model, id_name="id"):
        logging.info(f"Create if not exist with {model}")
        T = type(model)
        if not self.session.exec(select(T).where(getattr(T, id_name) == getattr(model, id_name))).first():
            self.session.add(model)
            self.session.commit()
        return model

    def add_model_template(self, model_template: ModelTemplateDB) -> int:
        # check if model template already exists
        existing_template = self.session.exec(
            select(ModelTemplateDB).where(ModelTemplateDB.name == model_template.name)
        ).first()
        if existing_template:
            logger.info(f"Model template with name {model_template.name} already exists. Returning existing id")
            return existing_template.id

        # add db entry
        logger.info(f"Adding model template: {model_template}")
        self.session.add(model_template)
        self.session.commit()

        # return id
        return model_template.id

    def add_model_template_from_yaml_config(self, model_template_config: ModelTemplateConfigV2) -> int:
        """Sets the ModelSpecRead a yaml string.
        Note that the yaml string is what's defined in a model template's MLProject file,
        so source_url will have to be added manually."""
        # TODO: maybe just use add_model_template and make sure to structure it correctly first
        # TODO: needs cleanup
        # TODO: existing check should probably use name instead of source url
        # parse yaml content as dict
        existing_template = self.session.exec(
            select(ModelTemplateDB).where(ModelTemplateDB.name == model_template_config.name)
        ).first()
        if existing_template:
            logger.info(f"Model template with name {model_template_config.name} already exists. Returning existing id")
            return existing_template.id
        d = model_template_config.dict()
        info = d.pop("meta_data")
        d = d | info
        db_object = ModelTemplateDB(**d)

        logger.info(f"Adding model template: {db_object}")
        self.session.add(db_object)
        self.session.commit()
        return db_object.id

    def add_configured_model(
        self, model_template_id: int, configuration: ModelConfiguration, configuration_name="default"
    ) -> int:
        # get model template name
        model_template = self.session.exec(select(ModelTemplateDB).where(ModelTemplateDB.id == model_template_id)).first()
        template_name = (
            model_template.name
        )

        # set configured name
        if configuration_name == "default":
            # default configurations just use the name of their model template (for backwards compatibility)
            name = template_name
        else:
            # combine model template with configuration name to make the name unique
            name = f"{template_name}:{configuration_name}"

        # check if configured model already exists
        existing_configured = self.session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.name == name)).first()
        if existing_configured:
            logger.info(f"Configured model with name {name} already exists. Returning existing id")
            return existing_configured.id

        # create and add db entry
        configured_model = ConfiguredModelDB(name=name, model_template_id=model_template_id, **configuration.dict(), model_template=model_template)
        configured_model.validate_user_options(configured_model)
        #configured_model.validate_user_options(model_template)
        logger.info(f"Adding configured model: {configured_model}")
        self.session.add(configured_model)
        self.session.commit()

        # return id
        return configured_model.id

    def get_configured_models(self) -> List[ModelSpecRead]:
        # TODO: using ModelSpecRead for backwards compatibility, should in future return ConfiguredModelDB?

        # get configured models from db
        # configured_models = SessionWrapper(session=session).list_all(ConfiguredModelDB)
        configured_models = self.session.exec(select(ConfiguredModelDB).join(ConfiguredModelDB.model_template)).all()

        # serialize to json and combine configured model with model template
        configured_models_data = []
        for configured_model in configured_models:
            # get configured model and model template json data
            configured_data = configured_model.model_dump(mode="json")
            template_data = configured_model.model_template.model_dump(mode="json")

            # add display name for configuration (not stored in db)
            # stitch together template displayName with configured name stub
            template_display_name = configured_model.model_template.display_name
            if ":" in configured_model.name:
                # configured model name is already stitched together as template_name:configuration_name
                configuration_stub = configured_model.name.split(":")[-1]
                # combine model template with configuration name to make the name unique
                configuration_display_name = configuration_stub.replace("_", " ").capitalize()
                display_name = f"{template_display_name} [{configuration_display_name}]"
            else:
                # default configurations just use the display name of their model template
                display_name = template_display_name
            configured_data["display_name"] = display_name

            # merge json data and add to results
            # NOTE: the sequence is important, starting with template data and add/overwrite with configured model data
            # ...in case of conflicting attrs, eg id and name
            merged_data = {**template_data, **configured_data}
            configured_models_data.append(merged_data)

        # debug
        # import json
        # for m in configured_models_data:
        #    logger.info('list model data: ' + json.dumps(m, indent=4))

        # temp: convert to ModelSpecRead to preserve existing results
        # TODO: remove ModelSpecRead and return directly as ConfiguredModelDB
        for model in configured_models_data:
            # convert single target value to target dict
            model["target"] = {
                "name": model["target"],
                "displayName": model["target"].replace("_", " ").capitalize(),
                "description": model["target"].replace("_", " ").capitalize(),
            }
            # convert list of required covarate strings to list of covariate dicts
            model["covariates"] = [
                {
                    "name": cov,
                    "displayName": cov.replace("_", " ").capitalize(),
                    "description": cov.replace("_", " ").capitalize(),
                }
                for cov in model["required_covariates"]
            ]
            # add list of additional covariate strings to list of covariate dicts
            model["covariates"] += [
                {
                    "name": cov,
                    "displayName": cov.replace("_", " ").capitalize(),
                    "description": cov.replace("_", " ").capitalize(),
                }
                for cov in model["additional_continuous_covariates"]
                if cov not in model["covariates"]
            ]
        # for m in configured_models_data:
        #    logger.info('converted list model data: ' + json.dumps(m, indent=4))
        configured_models_read = [ModelSpecRead.model_validate(m) for m in configured_models_data]
        # for m in configured_models_read:
        #    logger.info('read list model data: ' + json.dumps(m.model_dump(mode='json'), indent=4))

        # return
        return configured_models_read

    def get_configured_model_by_name(self, configured_model_name: str) -> ConfiguredModelDB:
        try:
            configured_model = self.session.exec(
            select(ConfiguredModelDB).where(ConfiguredModelDB.name == configured_model_name)
        ).one()
        except sqlalchemy.exc.NoResultFound:
            all_names = self.session.exec(
            select(ConfiguredModelDB.name)).all()
            raise ValueError(f"Configured model with name {configured_model_name} not found. Available names: {all_names}")

        return configured_model

    def get_configured_model_with_code(self, configured_model_id: int) -> ConfiguredModel:
        configured_model = self.session.get(ConfiguredModelDB, configured_model_id)
        if configured_model.name == "naive_model":
            return NaiveEstimator()
        template_name = configured_model.model_template.name
        ignore_env = template_name.startswith("chap_ewars") or template_name=='ewars_template'  # TODO: seems hacky, how to fix?
        return ModelTemplate.from_directory_or_github_url(
            configured_model.model_template.source_url,
            ignore_env=ignore_env,
        ).get_model(configured_model)

    def get_model_template(self, model_template_id: int) -> ModelTemplateInterface:
        model_template = self.session.get(ModelTemplateDB, model_template_id)
        if model_template is None:
            raise ValueError(f"Model template with id {model_template_id} not found")
        return model_template

    def add_evaluation_results(self, evaluation_results: Iterable[DataSet], last_train_period: TimePeriod, info: BackTestCreate):
        info.created = datetime.datetime.now()
        # org_units = list({location for ds in evaluation_results for location in ds.locations()})
        # split_points = list({er.period_range[0] for er in evaluation_results})
        model_db_id = self.session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.name == info.model_id)).first().id
        backtest = BackTest(**info.dict() | {'model_db_id': model_db_id})
        self.session.add(backtest)
        org_units = set([])
        split_points = set([])
        # define metrics (for each period)
        metric_defs = {
            'crps': crps_ensemble_timestep,
            'crps_norm': crps_ensemble_timestep_normalized,
            'is_within_10th_90th': is_within_10th_90th,
            'is_within_25th_75th': is_within_25th_75th,
        }
        # define aggregate metrics (for entire backtest)
        # value is tuple of (metric_id used to filter metric values, and function to run on filter metric values)
        aggregate_metric_defs = {
            'crps_mean': ('crps', lambda vals: np.mean(vals)),
            'crps_norm_mean': ('crps_norm', lambda vals: np.mean(vals)),
            'ratio_within_10th_90th': ('is_within_10th_90th', lambda vals: np.mean(vals)),
            'ratio_within_25th_75th': ('is_within_25th_75th', lambda vals: np.mean(vals)),
        }
        # begin loop
        evaluation_results = list(evaluation_results) # hacky, to avoid metric funcs using up the iterable before we can loop all splitpoints
        for eval_result in evaluation_results:
            first_period: TimePeriod = eval_result.period_range[0]
            split_points.add(first_period.id)
            for location, samples_with_truth in eval_result.items():
                # NOTE: samples_with_truth is class datatypes.SamplesWithTruth
                org_units.add(location)
                for period, sample_values, disease_cases in zip(eval_result.period_range, samples_with_truth.samples, samples_with_truth.disease_cases):
                    # add forecast series for this period
                    forecast = BackTestForecast(
                        period=period.id,
                        org_unit=location,
                        last_train_period=last_train_period.id,
                        last_seen_period=first_period.id,
                        values=sample_values.tolist(),
                    )
                    backtest.forecasts.append(forecast)
                    # add misc metrics
                    # TODO: should probably be improved with eg custom Metric classes
                    for metric_id, metric_func in metric_defs.items():
                        try:
                            metric_value = metric_func(sample_values, disease_cases, evaluation_results)
                            if np.isnan(metric_value) or np.isinf(metric_value):
                                logger.warning(f'Computed metric {metric_id} for location {location}, split period {first_period.id}, and forecast period {period.id} is NaN or Inf, skipping.')
                                continue

                            metric = BackTestMetric(
                                metric_id=metric_id, 
                                period=period.id, 
                                org_unit=location, 
                                last_train_period=last_train_period.id, 
                                last_seen_period=first_period.id, 
                                value=metric_value,
                            )
                            backtest.metrics.append(metric)
                        except Exception as err:
                            logger.warning(f'Unexpected error computing metric id {metric_id}, for location {location}, split period {first_period.id}, and forecast period {period.id}: {err}')
        # calculate and add total metrics
        # TODO: should probably be improved with eg custom Metric classes
        aggregate_metrics = {}
        for aggregate_metric_id, (filter_metric_id, aggregate_metric_func) in aggregate_metric_defs.items():
            try:
                filtered_metric_values = [
                    metric.value
                    for metric in backtest.metrics
                    if metric.metric_id == filter_metric_id
                ]
                aggregate_metric_value = float(aggregate_metric_func(filtered_metric_values))
                if np.isnan(aggregate_metric_value) or np.isinf(aggregate_metric_value):
                    logger.warning(f'Computed aggregate metric {aggregate_metric_id} is NaN or Inf, skipping.')
                    continue
                aggregate_metrics[aggregate_metric_id] = aggregate_metric_value
            except Exception as err:
                logger.warning(f'Unexpected error computing aggregate metric id {aggregate_metric_id}: {err}')
        logger.info(f'aggregate metrics {aggregate_metrics}')
        backtest.aggregate_metrics = aggregate_metrics
        # add more
        backtest.org_units = list(org_units)
        backtest.split_periods = list(split_points)
        self.session.commit()
        return backtest.id

    def add_predictions(self, predictions, dataset_id, model_id, name, metadata: dict = {}):
        n_periods = len(list(predictions.values())[0])
        prediction = Prediction(
            dataset_id=dataset_id,
            model_id=model_id,
            name=name,
            created=datetime.datetime.now(),
            n_periods=n_periods,
            meta_data=metadata,
            forecasts=[
                PredictionSamplesEntry(period=period.id, org_unit=location, values=value.tolist())
                for location, data in predictions.items()
                for period, value in zip(data.time_period, data.samples)
            ],
        )
        self.session.add(prediction)
        self.session.commit()
        return prediction.id

    def add_dataset(self, dataset_name, orig_dataset: _DataSet, polygons, dataset_type: str | None = None):
        logger.info(
            f"Adding dataset {dataset_name} with {len(list(orig_dataset.locations()))} locations and {len(orig_dataset.period_range)} time periods"
        )
        field_names = [
            field.name
            for field in dataclasses.fields(next(iter(orig_dataset.values())))
            if field.name not in ["time_period", "location"]
        ]
        logger.info(f"Field names in dataset: {field_names}")
        dataset = DataSet(
            name=dataset_name,
            polygons=polygons,
            created=datetime.datetime.now(),
            covariates=field_names,
            type=dataset_type,
        )
        for location, data in orig_dataset.items():
            field_names = [
                field.name for field in dataclasses.fields(data) if field.name not in ["time_period", "location"]
            ]
            for row in data:
                for field in field_names:
                    observation = Observation(
                        period=row.time_period.id,
                        org_unit=location,
                        value=float(getattr(row, field)),
                        feature_name=field,
                    )
                    dataset.observations.append(observation)
        self.session.add(dataset)
        self.session.commit()
        assert self.session.exec(select(Observation).where(Observation.dataset_id == dataset.id)).first() is not None
        return dataset.id

    def get_dataset(self, dataset_id, dataclass: type | None = None) -> _DataSet:
        dataset = self.session.get(DataSet, dataset_id)
        if dataclass is None:
            logger.info(f'Getting dataset with covariates: {dataset.covariates} and name: {dataset.name}')
            field_names = dataset.covariates
            dataclass = create_tsdataclass(field_names)
        observations = dataset.observations
        new_dataset = observations_to_dataset(dataclass, observations)
        return new_dataset

    def add_debug(self):
        """Function for debuging"""
        debug_entry = DebugEntry(timestamp=time.time())
        self.session.add(debug_entry)
        self.session.commit()
        return debug_entry.id


def create_db_and_tables():
    # TODO: Read config for options on how to create the database migrate/update/seed/seed_and_update
    if engine is not None:
        logger.info("Engin set. Creating tables")
        n = 0
        while n < 30:
            try:
                SQLModel.metadata.create_all(engine)
                break
            except sqlalchemy.exc.OperationalError as e:
                logger.error(f"Failed to create tables: {e}. Trying again")
                n += 1
                time.sleep(1)
        with SessionWrapper(engine) as session:
            from .model_template_seed import seed_configured_models_from_config_dir

            seed_configured_models_from_config_dir(session.session)
    else:
        logger.warning("Engine not set. Tables not created")
