import fakeredis
import pytest

from chap_core.rest_api.services.orchestrator import (
    DEFAULT_TTL_SECONDS,
    Orchestrator,
    ServiceNotFoundError,
)
from pydantic import ValidationError

from chap_core.rest_api.services.schemas import (
    MLServiceInfo,
    ModelMetadata,
    PeriodType,
    RegistrationRequest,
    ServiceInfo,
)


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def orchestrator(fake_redis):
    return Orchestrator(redis_client=fake_redis)


@pytest.fixture
def sample_payload():
    return RegistrationRequest(
        url="http://model-service:8080",
        info=MLServiceInfo(
            id="test-model",
            display_name="Test Model",
            model_metadata=ModelMetadata(author="Test Author"),
            period_type=PeriodType.monthly,
        ),
    )


def make_payload(service_id: str) -> RegistrationRequest:
    """Create a registration request with a specific service ID."""
    return RegistrationRequest(
        url=f"http://{service_id}:8080",
        info=MLServiceInfo(
            id=service_id,
            display_name=f"Service {service_id}",
            model_metadata=ModelMetadata(author="Test Author"),
            period_type=PeriodType.monthly,
        ),
    )


class TestRegister:
    def test_register_returns_registration_response(self, orchestrator, sample_payload):
        response = orchestrator.register(sample_payload)

        assert response.status == "registered"
        assert response.service_url == sample_payload.url
        assert response.ttl_seconds == DEFAULT_TTL_SECONDS
        assert response.id == sample_payload.info.id
        assert "$ping" in response.ping_url

    def test_register_stores_service_in_redis(self, orchestrator, sample_payload, fake_redis):
        response = orchestrator.register(sample_payload)

        key = f"service:{response.id}"
        assert fake_redis.exists(key)

    def test_reregister_updates_data(self, orchestrator, sample_payload):
        response1 = orchestrator.register(sample_payload)
        response2 = orchestrator.register(sample_payload)

        assert response1.id == response2.id
        assert response1.message == "Service registered successfully"
        assert response2.message == "Service registration updated"

    def test_reregister_with_new_url_updates_url(self, orchestrator):
        payload1 = make_payload("my-service")
        response1 = orchestrator.register(payload1)

        # Re-register with different URL
        payload2 = RegistrationRequest(
            url="http://new-url:9090",
            info=payload1.info,
        )
        response2 = orchestrator.register(payload2)

        assert response2.service_url == "http://new-url:9090"
        service = orchestrator.get("my-service")
        assert service.url == "http://new-url:9090"

    def test_different_ids_create_separate_registrations(self, orchestrator):
        response1 = orchestrator.register(make_payload("service-one"))
        response2 = orchestrator.register(make_payload("service-two"))

        assert response1.id != response2.id


class TestPing:
    def test_ping_updates_expiry(self, orchestrator, sample_payload):
        reg = orchestrator.register(sample_payload)
        original_expires = reg.ttl_seconds

        ping_response = orchestrator.ping(reg.id)

        assert ping_response.status == "alive"
        assert ping_response.id == reg.id

    def test_ping_nonexistent_service_raises_error(self, orchestrator):
        with pytest.raises(ServiceNotFoundError):
            orchestrator.ping("nonexistent-id")


class TestGet:
    def test_get_returns_service_detail(self, orchestrator, sample_payload):
        reg = orchestrator.register(sample_payload)

        service = orchestrator.get(reg.id)

        assert service.id == reg.id
        assert service.url == sample_payload.url
        assert service.info == sample_payload.info

    def test_get_nonexistent_service_raises_error(self, orchestrator):
        with pytest.raises(ServiceNotFoundError):
            orchestrator.get("nonexistent-id")


class TestGetAll:
    def test_get_all_returns_empty_list_initially(self, orchestrator):
        response = orchestrator.get_all()

        assert response.count == 0
        assert response.services == []

    def test_get_all_returns_registered_services(self, orchestrator):
        orchestrator.register(make_payload("service-one"))
        orchestrator.register(make_payload("service-two"))

        response = orchestrator.get_all()

        assert response.count == 2
        assert len(response.services) == 2


class TestDeregister:
    def test_deregister_removes_service(self, orchestrator, sample_payload, fake_redis):
        reg = orchestrator.register(sample_payload)
        key = f"service:{reg.id}"
        assert fake_redis.exists(key)

        orchestrator.deregister(reg.id)

        assert not fake_redis.exists(key)

    def test_deregister_nonexistent_service_raises_error(self, orchestrator):
        with pytest.raises(ServiceNotFoundError):
            orchestrator.deregister("nonexistent-id")


class TestServiceInfoSlugValidation:
    @pytest.mark.parametrize(
        "valid_id",
        [
            "my-service",
            "chap-ewars",
            "model1",
            "a",
            "abc123",
            "test-model-v2",
        ],
    )
    def test_valid_slug_ids(self, valid_id):
        info = ServiceInfo(id=valid_id, display_name="Test")
        assert info.id == valid_id

    @pytest.mark.parametrize(
        "invalid_id",
        [
            "Invalid-Slug",  # uppercase
            "123-service",  # starts with number
            "-my-service",  # starts with hyphen
            "my-service-",  # ends with hyphen
            "my--service",  # consecutive hyphens
            "my_service",  # underscore not allowed
            "my service",  # space not allowed
            "MY-SERVICE",  # all uppercase
            "",  # empty string
        ],
    )
    def test_invalid_slug_ids(self, invalid_id):
        with pytest.raises(ValidationError):
            ServiceInfo(id=invalid_id, display_name="Test")
