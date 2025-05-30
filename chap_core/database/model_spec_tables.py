import logging
from typing import Optional, List

from sqlalchemy import JSON, Column

from chap_core.database.base_tables import DBModel
from chap_core.database.feature_tables import FeatureType, ModelFeatureLink
from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData
from chap_core.model_spec import PeriodType
from sqlmodel import Field, Relationship

logger = logging.getLogger(__name__)


# TODO: legacy, most here will be outdated, and should be moved elsewhere


class ModelSpecBase(ModelTemplateMetaData, DBModel):
    """
    Use inheritance here so that it's flat in the database.
    """

    name: str
    # supported_period_types: PeriodType = PeriodType.any
    source_url: Optional[str] = None
    supported_period_type: PeriodType = PeriodType.any


class ModelSpecRead(ModelSpecBase):
    id: int
    covariates: List[FeatureType]
    target: FeatureType


target_type = FeatureType(name="disease_cases", display_name="Disease Cases", description="Disease Cases")


class ModelSpec(ModelSpecBase, table=True):
    """
    ModelSpec is the DB class for a Configured Model.
    It is configured through the "configuration" field which is JSON
    """

    id: Optional[int] = Field(primary_key=True, default=None)
    covariates: List[FeatureType] = Relationship(link_model=ModelFeatureLink)
    target_name: str = Field(foreign_key="featuretype.name")
    target: FeatureType = Relationship()
    configuration: Optional[dict] = Field(sa_column=Column(JSON))


def get_available_models_from_config_dir(config_dir: str, base_covariates) -> List[ModelSpec]:
    #  Reads from config dir, creates ModelSpec objects by reading from github_urls,
    # and returns a list of ModelSpec objects by calling ModelSpec.from_model_spec_read()
    pass


def get_available_models(base_covariates) -> List[ModelSpec]:
    """
    Returns a list of models that are available in chap
    """

    return [
        # in future, each of these can retrived by instead doing:
        # model_spec_read =  ModelSpecRead.from_github_url()
        # model_spec = ModelSpec.from_model_spec_read(model_spec_read)
        ModelSpec(
            name="naive_model",
            display_name="Naive model used for testing",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            description="A simple naive model only to be used for testing purposes.",
            supported_period_type=PeriodType.any,
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="NA",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Naive model used for testing". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="chap_ewars_weekly",
            display_name="Weekly CHAP-EWARS model",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            supported_period_type=PeriodType.week,
            description="Modified version of the World Health Organization (WHO) EWARS model. EWARS is a Bayesian hierarchical model implemented with the INLA library.",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Weekly CHAP-EWARS model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="chap_ewars_monthly",
            display_name="Monthly CHAP-EWARS",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            supported_period_type=PeriodType.month,
            description="Modified version of the World Health Organization (WHO) EWARS model. EWARS is a Bayesian hierarchical model implemented with the INLA library.",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/chap_auto_ewars@58d56f86641f4c7b09bbb635afd61740deff0640",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Monthly CHAP-EWARS model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="auto_regressive_weekly",
            display_name="Weekly Deep Auto Regressive",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            supported_period_type=PeriodType.week,
            description="An experimental deep learning model based on an RNN architecture, focusing on predictions based on auto-regressive time series data.",
            author="Knut Rand",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/knutdrand/weekly_ar_model@762ae74b7f972224bbea2f34e4e575cc127da8ea",
            contact_email="knut.rand@dhis2.org",
            citation_info='Rand, Knut. 2025. "Weekly Deep Auto Regressive model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="auto_regressive_monthly",
            display_name="Monthly Deep Auto Regressive",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            supported_period_type=PeriodType.month,
            description="An experimental deep learning model based on an RNN architecture, focusing on predictions based on auto-regressive time series data.",
            author="Knut Rand",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/monthly_ar_model@89f070dbe6e480d1e594e99b3407f812f9620d6d",
            contact_email="knut.rand@dhis2.org",
            citation_info='Rand, Knut. 2025. "Monthly Deep Auto Regressive model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
    ]
    """
        ModelSpec(
            name='madagascar_arima',
            parameters={},
            covariates=base_covariates,
            period=PeriodType.month,
            description="The Madagascar ARIMA model (with a wrapper), see https://github.com/dhis2-chap/Madagascar_ARIMA",
            author="Model by Michelle Evans, adapted by CHAP",
            organization="Pivot",
            source_url="https://github.com/dhis2-chap/Madagascar_ARIMA@a732bb4c88f36df8c8a07b11110b0db01170f8a0",
            target=target_type 
        ),
        ModelSpec(
            name='Epidemiar',
            parameters={},
            covariates=base_covariates,
            period=PeriodType.week,
            description="The Epidemiar model (adopted to fit with Chap, see https://github.com/dhis2-chap/epidemiar_example_model)",
            author="EcoGRAPH, adapted by CHAP",
            organization='EcoGRAPH',
            source_url="https://github.com/dhis2-chap/epidemiar_example_model@bc81de986cc139f90377005cb3b159307d1a359a",
            target=target_type
        ),
    ]
    """


def seed_with_session_wrapper(session_wrapper, get_models_func=get_available_models):
    """Seed a database using with the default models"""
    seeded_feature_types = [
        FeatureType(name="rainfall", display_name="Precipitation", description="Precipitation in mm"),
        FeatureType(
            name="mean_temperature", display_name="Mean Temperature", description="A measurement of mean temperature"
        ),
        FeatureType(name="population", display_name="Population", description="Population"),
        target_type,
    ]

    db_models = []
    for feature_type in seeded_feature_types:
        db_models.append(session_wrapper.create_if_not_exists(feature_type, id_name="name"))

    base_covariates = [db_models[0], db_models[1], db_models[2]]

    models = get_models_func(base_covariates)

    if models is not None:
        for model in models:
            session_wrapper.create_if_not_exists(model, id_name="name")
