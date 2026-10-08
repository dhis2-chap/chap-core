from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.models.builtin.registry import BUILTIN_SOURCE_PREFIX, BuiltinModel, BuiltinModelSpec, get_builtin_model
from chap_core.models.configured_model import ConfiguredModel, ModelConfiguration
from chap_core.models.model_template import ModelTemplate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def builtin_template_config(spec: BuiltinModelSpec) -> ModelTemplateConfigV2:
    """The model template config of a built-in model, as an MLproject file would give it."""
    return ModelTemplateConfigV2(
        name=spec.name,
        version=spec.version,
        source_url=f"{BUILTIN_SOURCE_PREFIX}{spec.name}",
        supported_period_type=spec.supported_period_type,
        required_covariates=list(spec.required_covariates),
        meta_data=ModelTemplateMetaData(
            display_name=spec.display_name,
            description=spec.description,
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            contact_email="chap@dhis2.org",
        ),
    )


class BuiltinConfiguredModel(ConfiguredModel):
    """Gives a built-in model the shape of an external model: ``train`` returns self and the model is callable."""

    def __init__(self, model: BuiltinModel, model_information: ModelTemplateConfigV2):
        self._model = model
        self._model_information = model_information

    @property
    def name(self) -> str:
        return self._model_information.name

    @property
    def model_information(self) -> ModelTemplateConfigV2:
        return self._model_information

    def train(self, train_data: DataSet, extra_args=None):
        self._model.train(train_data)
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        return self._model.predict(historic_data, future_data)

    def __call__(self):
        return self


class BuiltinModelTemplate(ModelTemplate):
    """Model template for a built-in model, resolved from the source URL ``builtin:<name>``."""

    def __init__(self, model_class: type[BuiltinModel]):
        super().__init__(builtin_template_config(model_class.spec), working_dir="")
        self._model_class = model_class

    @classmethod
    def from_source_url(cls, source_url: str) -> "BuiltinModelTemplate":
        return cls(get_builtin_model(source_url.removeprefix(BUILTIN_SOURCE_PREFIX)))

    def get_model(  # type: ignore[override]
        self,
        model_configuration: ModelConfiguration | None = None,
        prediction_length: int | None = None,
    ) -> BuiltinConfiguredModel:
        return BuiltinConfiguredModel(self._model_class(), self.model_template_config)
