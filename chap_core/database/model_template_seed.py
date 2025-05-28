from chap_core.model_spec import PeriodType
from chap_core.models.local_configuration import parse_local_model_config_from_directory

from .database import SessionWrapper
from .model_templates_and_config_tables import ModelTemplateDB, ModelConfiguration
from ..file_io.file_paths import get_config_path
from ..models.model_template import ExternalModelTemplate


def add_model_template(model_template: ModelTemplateDB, session_wrapper: SessionWrapper) -> int:
    template_id = session_wrapper.add_model_template(model_template)
    return template_id


def add_model_template_from_url(url: str, session_wrapper: SessionWrapper) -> int:
    model_template_config = ExternalModelTemplate.fetch_config_from_github_url(url)
    template_id = session_wrapper.add_model_template_from_yaml_config(model_template_config)
    return template_id


def add_configured_model(
    model_template_id, configuration: ModelConfiguration, configuration_name: str, session_wrapper: SessionWrapper
) -> int:
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
    return session_wrapper.add_configured_model(model_template_id, configuration, configuration_name)


def get_naive_model_template():
    model_template = ModelTemplateDB(
        name="naive_model",
        display_name="Naive model used for testing",
        required_covariates=["rainfall", "mean_temperature"],
        description="A simple naive model only to be used for testing purposes.",
        supported_period_type=PeriodType.any,
        author="CHAP team",
        author_assessed_status="gray",
        organization="HISP Centre, University of Oslo",
        organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
        source_url="NA",
        contact_email="chap@dhis2.org",
        citation_info='Climate Health Analytics Platform. 2025. "Naive model used for testing". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
    )
    return model_template


def seed_configured_models_from_config_dir(session, dir=get_config_path() / "configured_models"):
    wrapper = SessionWrapper(session=session)
    configured_models = parse_local_model_config_from_directory(dir)
    for config in configured_models:
        # for every version, add one for each configured model configuration
        for version, version_commit_or_branch in config.versions.items():
            version_commit_or_branch = version_commit_or_branch.strip("@")
            version_url = f"{config.url}@{version_commit_or_branch}"
            template_id = add_model_template_from_url(version_url, wrapper)
            for config_name, configured_model_configuration in config.configurations.items():
                add_configured_model(template_id, configured_model_configuration, config_name, wrapper)

    # add naive model template
    naive_template = get_naive_model_template()
    naive_template_id = add_model_template(naive_template, wrapper)
    # and naive configured model
    add_configured_model(
        naive_template_id,
        ModelConfiguration(additional_continuous_covariates=[], user_option_values={}),
        "default",
        wrapper,
    )
    session.commit()
