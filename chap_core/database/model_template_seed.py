from .database import SessionWrapper
from .model_spec_tables import ModelTemplateSpec
from ..models.model_template import ExternalModelTemplate

model_templates = [
    ('https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6')

]
template_urls = [
    'https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6',
    'https://github.com/sandvelab/monthly_ar_model@89f070dbe6e480d1e594e99b3407f812f9620d6d']



def add_model_template_from_url(url: str, session_wrapper: SessionWrapper) -> int:
    model_template_config = ExternalModelTemplate.fetch_config_from_github_url(url)
    template_id = session_wrapper.add_model_template(model_template_config)
    return template_id


def add_configured_model(model_template_id, configuration: dict, session_wrapper: SessionWrapper) -> int:
    """
    Add a configured model to the database.

    Parameters
    ----------
    model_template_id : int
        The ID of the model template.
    configuration : dict
        The configuration for the model.
    session_wrapper : SessionWrapper
        The session wrapper for database operations.

    Returns
    -------
    int
        The ID of the added model.
    """
    return session_wrapper.add_configured_model(model_template_id, configuration)


def model_templates() -> list[ModelTemplateSpec]:
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
            source_url="https://github.com/knutdrand/weekly_ar_model@762ae74b7f972224bbea2f34e4e575cc127da8ea",
            contact_email="knut.rand@dhis2.org",
            citation_info='Rand, Knut. 2025. "Weekly Deep Auto Regressive model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="auto_regressive_monthly",
            display_name='Monthly Deep Auto Regressive',
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
