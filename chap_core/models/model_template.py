from chap_core.datatypes import HealthData
from chap_core.external.model_configuration import ModelTemplateConfig
from chap_core.models.configured_model import ModelConfiguration
from chap_core.models.model_template_interface import ModelTemplateInterface
from chap_core.runners.runner import TrainPredictRunner


from pydantic import BaseModel, Field, create_model

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class ModelTemplate:
    """
    Represents a Model Template that can generate concrete models. 
    A template defines the choices allowed for a model
    """

    def __init__(self, model_template_config: ModelTemplateConfig, working_dir: str, ignore_env=False):
        self._model_template_config = model_template_config
        self._working_dir = working_dir
        self._ignore_env = ignore_env

    @property
    def name(self):
        return self._model_template_config.name

    def get_train_predict_runner(self) -> TrainPredictRunner:
        pass

    def __str__(self):
        return f'ModelTemplate: {self._model_template_config}'

    def get_config_class(self):
        fields = {}
        types = {'string': str, 'integer': int, 'float': float, 'boolean': bool}
        if self._model_template_config.allow_free_additional_continuous_covariates:
            fields["additional_continuous_covariates"] = (list[str], [])
        for user_option in self._model_template_config.user_options:
            T = types[user_option.type]
            if user_option.default is not None:
                fields[user_option.name] = (T, Field(default=T(user_option.default)))
            else:
                fields[user_option.name] = (T, ...)
        return create_model('ModelConfiguration', **fields)

    def get_default_model(self) -> 'ExternalModel':
        return self.get_model()

    def get_model(self, model_configuration: ModelConfiguration = None) -> 'ExternalModel':
        """
        Returns a model based on the model configuration. The model configuration is an object of the class
        returned by get_model_class (i.e. specified by the user). If no model configuration is passed, the default
        choices are used. 

        Parameters
        ----------
        model_configuration : ModelConfiguration, optional
            The configuration for the model, by default None

        Returns
        -------
        ExternalModel
            The model

        """
        # some choices are handled here, others are simply passedd on to the model
        config_passed_to_model = model_configuration

        # config = ModelTemplateConfig.model_validate(model_configuration)
        from chap_core.external.mlflow_wrappers import get_train_predict_runner_from_model_template_config
        from .external_model import ExternalModel
        runner = get_train_predict_runner_from_model_template_config(
            self._model_template_config,
            self._working_dir,
            self._ignore_env,
            model_configuration)

        config = self._model_template_config
        name = config.name
        adapters = config.adapters  # config.get("adapters", None)
        data_type = HealthData

        return ExternalModel(
            runner,
            name=name,
            adapters=adapters,
            data_type=data_type,
            working_dir=self._working_dir,
            configuration=config_passed_to_model
        )



class ExternalModelTemplate(ModelTemplateInterface):
    def __init__(self, model_template_config: ModelTemplateConfig, working_dir: str, ignore_env=False):
        self._model_template_config = model_template_config
        self._working_dir = working_dir
        self._ignore_env = ignore_env

    @classmethod
    def from_model_template_config(cls, model_template_config: ModelTemplateConfig, working_dir: str, ignore_env=False):
        return cls(ModelTemplate(model_template_config, working_dir, ignore_env))