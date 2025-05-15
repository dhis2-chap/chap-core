from chap_core.model_spec import PeriodType

from .database import SessionWrapper
from .model_templates_and_config_tables import ModelTemplateDB, ConfiguredModelDB
from ..models.model_template import ExternalModelTemplate

template_urls = {
    'https://github.com/sandvelab/monthly_ar_model@7c40890df749506c72748afda663e0e1cde4e36a': [{}],
    'https://github.com/knutdrand/weekly_ar_model@15cc39068498a852771c314e8ea989e6b555b8a5': [{}],
    'https://github.com/dhis2-chap/chap_auto_ewars@0c41b1d9bd187521e62c58d581e6f5bd5127f7b5': [{}],
    'https://github.com/dhis2-chap/chap_auto_ewars_weekly@51c63a8581bc29bdb40e788a83f701ed30cca83f': [{}],
}


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


def get_naive_model_spec():
    model_spec = ModelTemplateDB(
        name="naive_model",
        display_name='Naive model used for testing',
        required_covariates=['rainfall', 'mean_temperature'],
        description="A simple naive model only to be used for testing purposes.",
        supported_period_type=PeriodType.any,
        author="CHAP team",
        organization="HISP Centre, University of Oslo",
        organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
        source_url="NA",
        contact_email="chap@dhis2.org",
        citation_info='Climate Health Analytics Platform. 2025. "Naive model used for testing". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
    )
    return model_spec


def seed_configured_models(session):
    wrapper = SessionWrapper(session=session)
    for url, configs in template_urls.items():
        template_id = add_model_template_from_url(url, wrapper)
        for config in configs:
            add_configured_model(template_id, config, wrapper)
    spec = get_naive_model_spec()
    session.add(spec)
    session.commit()
    config = ConfiguredModelDB(name='default',
                               model_template_id=spec.id,
                               configuration={})
    session.add(config)
    session.commit()
    return spec
