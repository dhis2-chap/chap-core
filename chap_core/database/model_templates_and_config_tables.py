from typing import Optional, List
from enum import Enum

import jsonschema
from sqlalchemy import Column, JSON
from sqlmodel import SQLModel, Field, Relationship


from chap_core.database.base_tables import DBModel
from chap_core.model_spec import PeriodType
import logging
logger = logging.getLogger(__name__)


class AuthorAssessedStatus(Enum):
    gray = "gray"  # Gray: Not intended for use, or deprecated/meant for legacy use only.
    red = "red"  # Red: Highly experimental prototype - not at all validated and only meant for early experimentation
    orange = "orange"  # Orange: Has seen promise on limited data, needs manual configuration and careful evaluation
    yellow = "yellow"  # Yellow: Ready for more rigorous testing
    green = "green"  # Green: Validated, ready for use


class ModelTemplateMetaData(SQLModel):
    display_name: str = "No Display Name yet"
    description: str = "No Description yet"
    author_note: str = "No Author note yet"
    author_assessed_status: AuthorAssessedStatus = AuthorAssessedStatus("red")
    author: str = "Unknown Author"
    organization: Optional[str] = None
    organization_logo_url: Optional[str] = None
    contact_email: Optional[str] = None
    citation_info: Optional[str] = None


class ModelTemplateInformation(SQLModel):
    supported_period_type: PeriodType = PeriodType.any
    user_options: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    required_covariates: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    target: str = "disease_cases"
    allow_free_additional_continuous_covariates: bool = False


class ModelTemplateDB(DBModel, ModelTemplateMetaData, ModelTemplateInformation, table=True):
    """
    TODO: Maybe remove Spec from name, or find common convention for all models.
    Just a mixin here to get the model info flat in the database.
    """

    name: str = Field(unique=True)
    id: Optional[int] = Field(primary_key=True, default=None)
    source_url: Optional[str] = None


class ModelConfiguration(SQLModel):
    user_option_values: Optional[dict] = Field(sa_column=Column(JSON), default_factory=dict)
    additional_continuous_covariates: List[str] = Field(default_factory=list, sa_column=Column(JSON))

class ConfiguredModelDB(ModelConfiguration, DBModel, table=True):
    #  unique constraint on name
    # model_config = ConfigDict(protected_namespaces=())
    name: str = Field(unique=True)
    id: Optional[int] = Field(primary_key=True, default=None)
    model_template_id: int = Field(foreign_key="modeltemplatedb.id")
    model_template: ModelTemplateDB = Relationship()

    @classmethod
    def _validate_model_configuration(cls, user_options, user_option_values):
        logger.info(user_options)
        logger.info(user_option_values)
        schema = {
            "type": "object",
            "properties": user_options,
            "required": list({key for key, value in user_options.items() if "default" not in value}),
            "additionalProperties": False,
        }
        jsonschema.validate(instance=user_option_values, schema=schema)

    # @model_validator(mode='after')
    def validate_user_options(cls, model):
        try:
            cls._validate_model_configuration(model.model_template.user_options, model.user_option_values)
        except jsonschema.ValidationError as e:
            logger.error(f"Validation error in model configuration: {e}")
            raise ValueError(f"Invalid user options: {e.message}")
        return model
