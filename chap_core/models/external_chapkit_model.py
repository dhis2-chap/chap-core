import logging
from typing import Any

from chap_core.datatypes import Samples
from chap_core.exceptions import ModelFailedException
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.model_spec import PeriodType
from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper, RunInfo
from chap_core.models.chapkit_service_manager import ChapkitServiceManager, is_url
from chap_core.models.external_model import ExternalModelBase
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import TimePeriod

logger = logging.getLogger(__name__)

_CHAPKIT_PERIOD_MAP = {"weekly": "week", "monthly": "month"}


def _chapkit_period_to_chap(chapkit_period_type) -> PeriodType:
    """Map chapkit period type enum to CHAP PeriodType."""
    value = chapkit_period_type.value if hasattr(chapkit_period_type, "value") else str(chapkit_period_type)
    return PeriodType(_CHAPKIT_PERIOD_MAP.get(value, value))


def _parse_user_options_from_config_schema(config_schema: dict) -> dict:
    """Extract user-tuneable options from a chapkit config schema response.

    Legacy MLflow-era models store the schema under ``$defs.ModelConfiguration.properties``;
    chapkit services expose it at the top-level ``properties`` key. Filters out
    BaseConfig-reserved fields (``prediction_periods``, ``additional_continuous_covariates``)
    that chap-core handles separately.
    """
    user_options: dict = {}
    if "$defs" in config_schema and "ModelConfiguration" in config_schema["$defs"]:
        user_options = config_schema["$defs"]["ModelConfiguration"].get("properties", {})
    elif "properties" in config_schema:
        user_options = dict(config_schema["properties"])

    for reserved in ("prediction_periods", "additional_continuous_covariates"):
        user_options.pop(reserved, None)

    return user_options


def ml_service_info_to_model_template_config(
    info: "Any",
    rest_api_url: str,
    user_options: dict | None = None,
) -> ModelTemplateConfigV2:
    """Convert an MLServiceInfo object to a ModelTemplateConfigV2.

    Works with both chapkit's MLServiceInfo and the local schema copy
    in ``chap_core.rest_api.services.schemas``, since they share the
    same field names.
    """
    metadata = info.model_metadata
    meta_data_dict = {
        "display_name": info.display_name,
        "description": info.description or "No Description",
        "author_note": metadata.author_note or "",
        "author_assessed_status": metadata.author_assessed_status or "red",
        "author": metadata.author or "Unknown Author",
        "organization": metadata.organization,
        "organization_logo_url": str(metadata.organization_logo_url) if metadata.organization_logo_url else None,
        "contact_email": metadata.contact_email,
        "citation_info": metadata.citation_info,
        "documentation_url": str(metadata.documentation_url) if metadata.documentation_url else None,
    }

    config_dict = {
        "name": info.id,
        "version": info.version,
        "rest_api_url": rest_api_url,
        "meta_data": meta_data_dict,
        "required_covariates": info.required_covariates or [],
        "allow_free_additional_continuous_covariates": info.allow_free_additional_continuous_covariates,
        "requires_geo": getattr(info, "requires_geo", False),
        "supported_period_type": _chapkit_period_to_chap(info.period_type),
        "user_options": user_options or {},
        "min_prediction_periods": info.min_prediction_periods,
        "max_prediction_periods": info.max_prediction_periods,
        "entry_points": None,
        "docker_env": None,
        "python_env": None,
        "source_url": str(metadata.repository_url) if metadata.repository_url else rest_api_url,
    }

    return ModelTemplateConfigV2.model_validate(config_dict)


def _chapkit_config_payload(model_configuration: dict) -> dict:
    """The configuration data to store on the chapkit service.

    A chap-core configuration (a ``ModelConfiguration`` or a ``ConfiguredModelDB``
    row) carries bookkeeping fields the service has no schema for, so only the two
    fields that describe the configuration itself are sent. Anything else is a raw
    dict authored against the service's own config schema, and is passed through
    minus the ``model_template`` key chap-core attaches.

    An empty ``additional_continuous_covariates`` is dropped rather than sent: the
    row defaults it to an empty list, which would override whatever default the
    service's own schema declares (ewars, for instance, defaults to rainfall and
    mean_temperature).
    """
    covariates = model_configuration.get("additional_continuous_covariates")
    if "user_option_values" in model_configuration:
        payload = {}
        if model_configuration.get("user_option_values"):
            payload["user_option_values"] = model_configuration["user_option_values"]
    else:
        payload = {key: value for key, value in model_configuration.items() if key != "model_template"}
        payload.pop("additional_continuous_covariates", None)
    if covariates:
        payload["additional_continuous_covariates"] = covariates
    return payload


class ExternalChapkitModelTemplate:
    """Wrapper around External models that are based on chapkit.

    Note that get_model assumes you have already created a configuration with that specific chapkitmodel.

    This class is meant to be backwards compatible with ExternalModelTemplate.

    Supports two modes:
    1. URL mode (backwards compatible): Pass a URL to a running service
    2. Directory mode (new): Pass a path to a model directory, service
       will be auto-started when used as context manager

    Example usage (directory mode):
        with ExternalChapkitModelTemplate("/path/to/model") as template:
            model = template.get_model({})
            model.train(data)
            predictions = model.predict(historic, future)

    Example usage (URL mode - backwards compatible):
        template = ExternalChapkitModelTemplate("http://localhost:8000")
        template.wait_for_healthy()
        model = template.get_model({})
    """

    def __init__(
        self,
        path_or_url: str,
        port: int | None = None,
        host: str = "127.0.0.1",
        startup_timeout: int = 60,
    ):
        """
        Initialize the template.

        Args:
            path_or_url: Either a URL to a running service, or a path to
                        a model directory
            port: Port to use when auto-starting (directory mode only)
            host: Host to bind to when auto-starting (default: 127.0.0.1)
            startup_timeout: Seconds to wait for service startup
        """
        self._path_or_url = path_or_url
        self._port = port
        self._host = host
        self._startup_timeout = startup_timeout

        self._is_url_mode = is_url(path_or_url)

        if self._is_url_mode:
            self.rest_api_url: str | None = path_or_url
            self.client: CHAPKitRestAPIWrapper | None = CHAPKitRestAPIWrapper(path_or_url)
            self._service_manager: ChapkitServiceManager | None = None
        else:
            self.rest_api_url = None
            self.client = None
            self._service_manager = ChapkitServiceManager(
                model_directory=path_or_url,
                port=port,
                host=host,
                startup_timeout=startup_timeout,
            )

    @property
    def is_url_mode(self) -> bool:
        return self._is_url_mode

    def _ensure_initialized(self) -> None:
        """Ensure the template is initialized (service running, client ready)."""
        if self.client is None:
            raise RuntimeError(
                "Template not initialized. When using directory mode, "
                "use as context manager: with ExternalChapkitModelTemplate(path) as template: ..."
            )

    def __enter__(self) -> "ExternalChapkitModelTemplate":
        """Start service if in directory mode."""
        if self._is_url_mode:
            return self

        assert self._service_manager is not None
        self._service_manager.__enter__()
        self.rest_api_url = self._service_manager.url
        self.client = CHAPKitRestAPIWrapper(self.rest_api_url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the client, and stop the service if in directory mode."""
        if self.client is not None:
            self.client.close()
        if self._service_manager is not None and not self._is_url_mode:
            self._service_manager.__exit__(exc_type, exc_val, exc_tb)
            self.rest_api_url = None
            self.client = None

    def wait_for_healthy(self, timeout=60):
        self._ensure_initialized()
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_healthy():
                return True
            logger.info("Waiting for model service to become healthy...")
            time.sleep(2)
        raise TimeoutError(
            f"Model service at {self.rest_api_url} did not become healthy within {timeout} seconds. Check {self.rest_api_url}/health"
        )

    def is_healthy(self) -> bool:
        self._ensure_initialized()
        assert self.client is not None
        try:
            response = self.client.health()
            return response.status == "healthy"
        except Exception as e:
            logger.info(
                f"Health check for model {self.rest_api_url} failed: {e}. Check health at {self.rest_api_url}/health"
            )
            return False

    def get_model(
        self,
        model_configuration: dict,
        prediction_length: int | None = None,
    ) -> "ExternalChapkitModel":
        """
        Sends the model configuration for storing in the model (by sending to the model rest api).
        This returns a configuration id back that we can use to identify the model.

        ``prediction_length`` is the forecast horizon the model will be asked for.
        It is not part of the stored configuration; the model carries it and sends
        it as ``run_info.prediction_periods`` at train time. The name matches
        ``ModelTemplate.get_model`` so both template kinds share one interface.
        """
        self._ensure_initialized()
        assert self.client is not None
        assert self.rest_api_url is not None
        import time

        model_configuration = {} if model_configuration is None else dict(model_configuration)

        timestamp = int(time.time() * 1000000)
        if model_configuration.get("name"):
            # always make sure config has unique name for now. Chapkit uses name as identifier,
            # but we don't necesserarily do that on the chap side
            name = f"{model_configuration['name']}_{timestamp}"
        else:
            name = f"{self.name}_config_{timestamp}"

        config_data = {"name": name, "data": _chapkit_config_payload(model_configuration)}
        logger.info(f"Creating model configuration with name {name} at {self.rest_api_url}. Data: {config_data}")

        config_response = self.client.create_config(config_data)
        configuration_id = str(config_response.id)

        # get all configs and assert that configuration_id is there
        all_configs = self.client.list_configs()
        assert any(cfg.id == configuration_id for cfg in all_configs), (
            f"Created configuration {configuration_id} not found in list of configs"
        )

        logger.info(f"Created model configuration with id {configuration_id} at {self.rest_api_url}")
        return ExternalChapkitModel(
            self.name,
            self.rest_api_url,
            configuration_id=configuration_id,
            model_information=self.model_template_config,
            client=self.client,
            prediction_periods=prediction_length,
        )

    @property
    def name(self):
        """Return the chapkit service id as the model template name."""
        self._ensure_initialized()
        assert self.client is not None
        info = self.client.info()
        return info.id

    @property
    def model_template_config(self) -> ModelTemplateConfigV2:
        """Property alias for get_model_template_config for backwards compatibility."""
        return self.get_model_template_config()

    def get_model_template_config(self) -> ModelTemplateConfigV2:
        """
        This method is meant to make things backwards compatible with old system. An object of type
        ModelTemplateConfigV2 is needed to store info about a ModelTemplate in the database.
        """
        return self.get_model_template_config_with_digest()[0]

    def get_model_template_config_with_digest(self) -> tuple[ModelTemplateConfigV2, str | None]:
        """Fetch the service info once and return the template config together with the commit
        the service reports it was built from, or None when it does not report a usable one."""
        self._ensure_initialized()
        assert self.client is not None
        assert self.rest_api_url is not None
        model_info = self.client.info()

        config_schema = self.client.get_config_schema()
        user_options = _parse_user_options_from_config_schema(config_schema)

        config = ml_service_info_to_model_template_config(model_info, self.rest_api_url, user_options)
        return config, model_info.git_revision


class ExternalChapkitModel(ExternalModelBase):
    def __init__(
        self,
        model_name: str,
        rest_api_url: str,
        configuration_id: str,
        model_information: ModelTemplateConfigV2 | None = None,
        client: CHAPKitRestAPIWrapper | None = None,
        prediction_periods: int | None = None,
    ):
        self.model_name = model_name
        self.rest_api_url = rest_api_url
        self.configuration_id = configuration_id
        self._location_mapping = None
        self._adapters = None
        # Share the template's client when given so one connection pool serves
        # both, instead of opening a second one that nothing closes.
        self.client = client if client is not None else CHAPKitRestAPIWrapper(rest_api_url)
        self._train_id: str | None = None
        self._model_information = model_information
        self._prediction_periods = prediction_periods

    @property
    def model_information(self):
        return self._model_information

    def _train_horizon(self) -> int | None:
        """The horizon to request at train time, clamped to the model's declared bounds.

        ``None`` means no horizon was requested, which leaves the key out of the
        request so the service keeps the horizon from its stored configuration.
        A requested horizon above the model's maximum is clamped because CHAP wraps
        such models in :class:`ExtendedPredictor` and asks them for at most their
        maximum per prediction, so that is what they should be trained for. chapkit
        rejects a horizon outside the declared bounds.
        """
        requested = self._prediction_periods
        if requested is None:
            return None
        info = self.model_information
        if info is None:
            return requested
        horizon = requested
        if info.min_prediction_periods is not None:
            horizon = max(horizon, info.min_prediction_periods)
        if info.max_prediction_periods is not None:
            horizon = min(horizon, info.max_prediction_periods)
        if horizon != requested:
            logger.warning(
                "Requested prediction horizon %d is outside the bounds declared by %s "
                "(min %s, max %s); training for %d periods instead",
                requested,
                self.model_name,
                info.min_prediction_periods,
                info.max_prediction_periods,
                horizon,
            )
        return horizon

    def train(self, train_data: DataSet, extra_args=None, run_info: RunInfo | None = None):
        frequency = self._get_frequency(train_data)
        df = train_data.to_pandas()
        new_df = self._adapt_data(df, frequency=frequency)
        geo = train_data.polygons
        if run_info is None:
            run_info = RunInfo(prediction_periods=self._train_horizon())
        job, artifact_id = self.client.train_and_wait(self.configuration_id, new_df, run_info, geo)

        if job.status != "completed":
            raise ModelFailedException(
                f"Training job {job.id} ended with status '{job.status}': {job.error or 'Unknown error'}. "
                f"Stacktrace: {job.error_traceback or ''}"
            )

        assert artifact_id is not None, f"No artifact_id returned: {job}"
        self._train_id = artifact_id
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet, run_info: RunInfo | None = None) -> DataSet:
        assert self._train_id is not None, "Model must be trained before prediction"
        geo = historic_data.polygons
        historic_data_pd = self._adapt_data(historic_data.to_pandas())
        future_data_pd = self._adapt_data(future_data.to_pandas())
        if run_info is None:
            run_info = RunInfo(prediction_periods=len(future_data.period_range))
        job, artifact_id = self.client.predict_and_wait(
            artifact_id=self._train_id,
            future_data=future_data_pd,
            run_info=run_info,
            historic_data=historic_data_pd,
            geo_features=geo,
        )

        if job.status != "completed":
            raise ModelFailedException(
                f"Prediction job {job.id} ended with status '{job.status}': {job.error or 'Unknown error'}. "
                f"Stacktrace: {job.error_traceback or ''}"
            )

        assert artifact_id is not None, f"No prediction artifact: {job.error or ''}"

        # get artifact from the client
        prediction_data = self.client.get_prediction_artifact_dataframe(artifact_id)
        pd = prediction_data.to_pandas()
        # Drop any predictions before the future window start. Some models (e.g. EWARS)
        # run inference over the combined historic+future frame and emit predictions for
        # every row with NA disease_cases, which includes missing-reporting months in
        # historic data. Mirrors the legacy path in ExternalCommandLineModel.predict.
        if "time_period" in pd.columns:
            start_time = future_data.start_timestamp
            time_periods = [TimePeriod.parse(s) for s in pd.time_period.astype(str)]
            mask = [start_time <= tp.start_timestamp for tp in time_periods]
            pd = pd[mask]
        if "time_period" in pd.columns and "location" in pd.columns:
            pd = pd.sort_values(by=["location", "time_period"]).reset_index(drop=True)
        return DataSet.from_pandas(pd, Samples)
