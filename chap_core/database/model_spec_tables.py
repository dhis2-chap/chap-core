import logging
from typing import Optional, List

import requests
from sqlalchemy import JSON, Column
import yaml

from chap_core.database.base_tables import DBModel
from chap_core.model_spec import PeriodType
from sqlmodel import Field, Relationship

from chap_core.util import parse_github_url
logger = logging.getLogger(__name__)


class FeatureTypeBase(DBModel):
    display_name: str
    description: str


class FeatureTypeRead(FeatureTypeBase):
    name: str


class FeatureType(FeatureTypeBase, table=True):
    name: str = Field(str, primary_key=True)


class FeatureSource(DBModel, table=True):
    name: str = Field(primary_key=True)
    display_name: str
    feature_type: str = Field(foreign_key="featuretype.name")
    provider: str
    supported_period_types: List[PeriodType] = Field(default_factory=list, sa_column=Column(JSON))


class ModelFeatureLink(DBModel, table=True):
    model_id: Optional[int] = Field(default=None, foreign_key="modelspec.id", primary_key=True)
    feature_type: Optional[str] = Field(default=None, foreign_key="featuretype.name", primary_key=True)


class ModelSpecBase(DBModel):
    name: str
    display_name: str
    supported_period_types: PeriodType = PeriodType.any
    description: str = "No Description yet"
    author: str = "Unknown Author"
    organization: Optional[str] = None
    organization_logo_url: Optional[str] = None
    source_url: Optional[str] = None
    contact_email: Optional[str] = None
    citation_info: Optional[str] = None


class ModelSpecRead(ModelSpecBase):
    id: int
    covariates: List[FeatureTypeRead]
    target: FeatureTypeRead

    @classmethod
    def from_github_url(cls, github_url: str) -> 'ModelSpecRead':
        # parse the github url
        parsed = parse_github_url(github_url)
        # Takes a github url, parses the MLProject file, returns an object with the correct information
        raw_mlproject_url = f"https://raw.githubusercontent.com/{parsed.owner}/{parsed.repo_name}/{parsed.commit}/MLproject"
        # fetch this MLProject file and parse it
        try: 
            fetched = requests.get(raw_mlproject_url)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MLProject file: {e}")
            return None
        content = fetched.content
        # parse content as yaml
        content = yaml.safe_load(content)
        
        # todo:
        # parse as much as possible from the MLProject file (but allow missing data)

        return cls(
            id = 0,
            source_url = github_url,
            name = content["name"],
            covariates=[],
            target=FeatureTypeRead(name='disease_cases', display_name='Disease Cases', description='Disease Cases'),
            display_name = content["name"],
            description = content.get("description", "No Description yet"),
        )

 
target_type = FeatureType(name='disease_cases',
                          display_name='Disease Cases',
                          description='Disease Cases')


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
           
    @classmethod
    def from_model_spec_read(cls, base_covariates, model_spec_read) -> 'ModelSpec':
        return cls(
            name=model_spec_read.name,
            display_name=model_spec_read.display_name,
            supported_period_types=model_spec_read.supported_period_types,
            description=model_spec_read.description,
            author=model_spec_read.author,
            organization=model_spec_read.organization,
            organization_logo_url=model_spec_read.organization_logo_url,
            source_url=model_spec_read.source_url,
            contact_email=model_spec_read.contact_email,
            citation_info=model_spec_read.citation_info,
            covariates=base_covariates,
            target=target_type
        )



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
            display_name='Naive model used for testing',
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.month,
            description="A simple naive model only to be used for testing purposes.",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="NA",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Naive model used for testing". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="chap_ewars_weekly",
            display_name='Weekly CHAP-EWARS model',
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.week,
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
            display_name='Monthly CHAP-EWARS',
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.month,
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
            display_name='Weekly Deep Auto Regressive',
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.week,
            description="An experimental deep learning model based on an RNN architecture, focusing on predictions based on auto-regressive time series data.",
            author="Knut Rand",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/knutdrand/weekly_ar_model@1730b26996201d9ee0faf65695f44a2410890ea5",
            contact_email="knut.rand@dhis2.org",
            citation_info='Rand, Knut. 2025. "Weekly Deep Auto Regressive model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="auto_regressive_monthly",
            displayName='Monthly Deep Auto Regressive',
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.month,
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
    '''Seed a database using with the default models'''
    seeded_feature_types = [
        FeatureType(name='rainfall',
                    display_name='Precipitation',
                    description='Precipitation in mm'),
        FeatureType(name='mean_temperature',
                    display_name='Mean Temperature',
                    description='A measurement of mean temperature'),
        FeatureType(name='population',
                    display_name='Population',
                    description='Population'),
        target_type]

    db_models = []
    for feature_type in seeded_feature_types:
        db_models.append(session_wrapper.create_if_not_exists(feature_type, id_name='name'))

    base_covariates = [db_models[0], db_models[1], db_models[2]]

    models = get_models_func(base_covariates)
    
    if models is not None:
        for model in models:
            session_wrapper.create_if_not_exists(model, id_name='name')
