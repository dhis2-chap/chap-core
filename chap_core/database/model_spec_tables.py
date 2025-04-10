from typing import Optional, List

from sqlalchemy import JSON, Column

from chap_core.database.base_tables import DBModel
from chap_core.model_spec import PeriodType
from sqlmodel import Field, Relationship


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


# TODO: Move to db spec
# ModelTags = Literal["bayesian", "deep learning"]

class ModelSpecBase(DBModel):
    name: str  # TODO: add nameLong field to contain human readable version, so that description field can contain longer text
    supported_period_types: PeriodType = PeriodType.any
    description: str = "No Description yet"
    author: str = "Unknown Author"
    organization: Optional[str] = None
    author_logo_url: Optional[str] = None  # TODO: rename to organization_logo_url 
    source_url: Optional[str] = None
    contact_email: Optional[str] = None
    citation_info: Optional[str] = None

class ModelSpecRead(ModelSpecBase):
    id: int
    covariates: List[FeatureTypeRead]
    target: FeatureTypeRead


class ModelSpec(ModelSpecBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    covariates: List[FeatureType] = Relationship(link_model=ModelFeatureLink)  # TODO: rename to covariates
    target_name: str = Field(foreign_key="featuretype.name")  # TODO: rename to name
    target: FeatureType = Relationship()


target_type = FeatureType(name='disease_cases',
                          display_name='Disease Cases',
                          description='Disease Cases')


def seed_with_session_wrapper(session_wrapper):
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

    seeded_models = [
        ModelSpec(
            name="naive_model",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.month,
            description="Naive model used for testing",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            author_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="NA",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Naive model used for testing". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="chap_ewars_weekly",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.week,
            description="Weekly CHAP-EWARS model",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            author_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Weekly CHAP-EWARS model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="chap_ewars_monthly",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.month,
            description="Monthly CHAP-EWARS model",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            author_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/chap_auto_ewars@58d56f86641f4c7b09bbb635afd61740deff0640",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Monthly CHAP-EWARS model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="auto_regressive_weekly",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.week,
            description="Weekly Deep Auto Regressive model",
            author="Knut Rand",
            organzation="HISP Centre, University of Oslo",
            author_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/knutdrand/weekly_ar_model@36a537dac138af428a4167b2a89eac7dafd5d762",
            contact_email="knut.rand@dhis2.org",
            citation_info='Rand, Knut. 2025. "Weekly Deep Auto Regressive model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="auto_regressive_monthly",
            parameters={},
            target=target_type,
            covariates=base_covariates,
            period=PeriodType.month,
            description="Monthly Deep Auto Regressive model",
            author="Knut Rand",
            organzation="HISP Centre, University of Oslo",
            author_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/monthly_ar_model@cadd785872624b4bcd839a39f5e7020c25254c31",
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

    for model in seeded_models:
        session_wrapper.create_if_not_exists(model, id_name='name')

