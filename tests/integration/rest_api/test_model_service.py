import fakeredis
import pytest
from unittest.mock import MagicMock, patch

from chap_core.database.feature_tables import FeatureType
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.rest_api.services.model_service import ModelService, _service_to_model_spec
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.services.schemas import RegistrationPayload, ServiceDetail


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def orchestrator(fake_redis):
    return Orchestrator(redis_client=fake_redis)


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def sample_static_model():
    return ModelSpecRead(
        id=1,
        name="static_model",
        display_name="Static Model",
        description="A static model from the database",
        author="Test Author",
        source_url="https://example.com/static",
        covariates=[],
        target=FeatureType(
            name="disease_cases",
            display_name="Disease Cases",
            description="Disease Cases",
        ),
        archived=False,
    )


@pytest.fixture
def sample_registration_payload():
    return RegistrationPayload(
        url="http://model-service:8080",
        info={
            "name": "dynamic-model",
            "display_name": "Dynamic Model",
            "description": "A dynamically registered model",
            "author": "Dynamic Author",
        },
    )


class TestModelService:
    def test_get_all_models_returns_static_models(self, mock_session, orchestrator, sample_static_model):
        with patch.object(ModelService, "_get_static_models", return_value=[sample_static_model]):
            service = ModelService(session=mock_session, orchestrator=orchestrator)
            models = service.get_all_models()

            assert len(models) == 1
            assert models[0].name == "static_model"

    def test_get_all_models_returns_dynamic_models(self, mock_session, orchestrator, sample_registration_payload):
        orchestrator.register(sample_registration_payload)

        with patch.object(ModelService, "_get_static_models", return_value=[]):
            service = ModelService(session=mock_session, orchestrator=orchestrator)
            models = service.get_all_models()

            assert len(models) == 1
            assert models[0].name == "dynamic-model"
            assert models[0].id == -1  # Dynamic models have negative ID

    def test_get_all_models_combines_static_and_dynamic(
        self, mock_session, orchestrator, sample_static_model, sample_registration_payload
    ):
        orchestrator.register(sample_registration_payload)

        with patch.object(ModelService, "_get_static_models", return_value=[sample_static_model]):
            service = ModelService(session=mock_session, orchestrator=orchestrator)
            models = service.get_all_models()

            assert len(models) == 2
            # Static models come first
            assert models[0].name == "static_model"
            assert models[0].id == 1
            # Dynamic models come second
            assert models[1].name == "dynamic-model"
            assert models[1].id == -1

    def test_get_dynamic_models_returns_empty_when_redis_fails(self, mock_session):
        # Create orchestrator with a mock that raises an exception
        mock_orchestrator = MagicMock()
        mock_orchestrator.get_all.side_effect = Exception("Redis connection failed")

        service = ModelService(session=mock_session, orchestrator=mock_orchestrator)
        models = service._get_dynamic_models()

        assert models == []

    def test_get_dynamic_models_skips_failed_conversions(self, mock_session, orchestrator):
        # Register two services
        orchestrator.register(
            RegistrationPayload(
                url="http://service1:8080",
                info={"name": "service1"},
            )
        )
        orchestrator.register(
            RegistrationPayload(
                url="http://service2:8080",
                info={"name": "service2"},
            )
        )

        service = ModelService(session=mock_session, orchestrator=orchestrator)

        # Mock _service_to_model_spec to fail on first service
        with patch("chap_core.rest_api.services.model_service._service_to_model_spec") as mock_convert:
            mock_convert.side_effect = [
                Exception("Conversion failed"),
                ModelSpecRead(
                    id=-1,
                    name="service2",
                    display_name="service2",
                    description="test",
                    author="Unknown",
                    covariates=[],
                    target=FeatureType(
                        name="disease_cases",
                        display_name="Disease Cases",
                        description="Disease Cases",
                    ),
                    archived=False,
                ),
            ]

            models = service._get_dynamic_models()

            # Only the second service should be in the list
            assert len(models) == 1
            assert models[0].name == "service2"


class TestServiceToModelSpec:
    def test_converts_service_with_full_info(self):
        service = ServiceDetail(
            id="test-ulid",
            url="http://model:8080",
            info={
                "name": "my-model",
                "display_name": "My Model",
                "description": "A test model",
                "author": "Test Author",
            },
            registered_at="2024-01-01T00:00:00Z",
            last_updated="2024-01-01T00:00:00Z",
            last_ping_at="2024-01-01T00:00:00Z",
            expires_at="2024-01-01T00:00:30Z",
        )

        model = _service_to_model_spec(service)

        assert model.id == -1
        assert model.name == "my-model"
        assert model.display_name == "My Model"
        assert model.description == "A test model"
        assert model.author == "Test Author"
        assert model.source_url == "http://model:8080"
        assert model.archived is False

    def test_uses_defaults_for_missing_info(self):
        service = ServiceDetail(
            id="test-ulid",
            url="http://model:8080",
            info={},  # Empty info
            registered_at="2024-01-01T00:00:00Z",
            last_updated="2024-01-01T00:00:00Z",
            last_ping_at="2024-01-01T00:00:00Z",
            expires_at="2024-01-01T00:00:30Z",
        )

        model = _service_to_model_spec(service)

        assert model.id == -1
        assert model.name == "test-ulid"  # Falls back to service ID
        assert model.display_name == "test-ulid"  # Falls back to name
        assert model.description == "Dynamically registered chapkit service"
        assert model.author == "Unknown"

    def test_uses_service_id_when_name_missing(self):
        service = ServiceDetail(
            id="my-service-ulid",
            url="http://model:8080",
            info={"display_name": "Custom Display"},
            registered_at="2024-01-01T00:00:00Z",
            last_updated="2024-01-01T00:00:00Z",
            last_ping_at="2024-01-01T00:00:00Z",
            expires_at="2024-01-01T00:00:30Z",
        )

        model = _service_to_model_spec(service)

        assert model.name == "my-service-ulid"
        assert model.display_name == "Custom Display"
