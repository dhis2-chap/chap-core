"""
Service for retrieving all available models from both static and dynamic sources.

Static models are stored in the database (ConfiguredModelDB).
Dynamic models come from chapkit services registered via the v2 service registration API.
"""

import logging
from typing import Any

from sqlmodel import Session

from chap_core.database.database import SessionWrapper
from chap_core.database.feature_tables import FeatureType
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.services.schemas import ServiceDetail

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for retrieving all available models, combining static database
    models with dynamically registered chapkit services.

    Static models come from the database (ConfiguredModelDB).
    Dynamic models come from chapkit services registered via the v2
    service registration API.
    """

    def __init__(self, session: Session, orchestrator: Orchestrator):
        """
        Initialize the ModelService.

        Args:
            session: Database session for accessing static models.
            orchestrator: Orchestrator instance for accessing registered services.
        """
        self._session = session
        self._orchestrator = orchestrator

    def get_all_models(self) -> list[ModelSpecRead]:
        """
        Retrieve all available models from both static and dynamic sources.

        Returns:
            Combined list of ModelSpecRead, with static models first,
            followed by dynamic models from registered services.
        """
        static_models = self._get_static_models()
        dynamic_models = self._get_dynamic_models()
        return static_models + dynamic_models

    def _get_static_models(self) -> list[ModelSpecRead]:
        """Retrieve models from database."""
        session_wrapper = SessionWrapper(session=self._session)
        return session_wrapper.get_configured_models()

    def _get_dynamic_models(self) -> list[ModelSpecRead]:
        """
        Convert registered chapkit services to ModelSpecRead format.

        Uses service info to populate model metadata. Services that
        fail to convert are logged and skipped.

        Returns:
            List of ModelSpecRead for each registered service.
            Returns empty list if Redis is unavailable.
        """
        try:
            services = self._orchestrator.get_all()
            models = []
            for service in services.services:
                try:
                    models.append(_service_to_model_spec(service))
                except Exception as e:
                    logger.warning(f"Failed to convert service {service.id} to model: {e}")
            return models
        except Exception as e:
            logger.warning(f"Failed to get dynamic models: {e}")
            return []


def _service_to_model_spec(service: ServiceDetail) -> ModelSpecRead:
    """
    Convert a registered service to ModelSpecRead format.

    Args:
        service: ServiceDetail from orchestrator containing chapkit service info.

    Returns:
        ModelSpecRead with fields populated from service info.
        Uses defaults for most fields as the info schema will be
        standardized in a future PR.
    """
    info: dict[str, Any] = service.info
    name = info.get("name", service.id)

    return ModelSpecRead(
        id=-1,  # Negative ID indicates dynamic model
        name=name,
        display_name=info.get("display_name", name),
        description=info.get("description", "Dynamically registered chapkit service"),
        author=info.get("author", "Unknown"),
        source_url=service.url,
        covariates=[],  # Empty for now, will be standardized later
        target=FeatureType(
            name="disease_cases",
            display_name="Disease Cases",
            description="Disease Cases",
        ),
        archived=False,
    )
