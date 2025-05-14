from .database import SessionWrapper
from .model_spec_tables import ModelTemplateSpec
from ..models.model_template import ExternalModelTemplate

template_urls = {
    'https://github.com/sandvelab/monthly_ar_model@7c40890df749506c72748afda663e0e1cde4e36a': [{}],
    'https://github.com/knutdrand/weekly_ar_model@15cc39068498a852771c314e8ea989e6b555b8a5': [{}]
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


def seed_configured_models(session_wrapper):
    for url, configs in template_urls.items():
        template_id = add_model_template_from_url(url, session_wrapper)
        for config in configs:
            add_configured_model(template_id, config, session_wrapper)