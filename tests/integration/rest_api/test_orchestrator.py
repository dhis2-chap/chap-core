import fakeredis
import pytest

from chap_core.rest_api.services.orchestrator import (
    DEFAULT_TTL_SECONDS,
    Orchestrator,
    ServiceNotFoundError,
)
from chap_core.rest_api.services.schemas import RegistrationPayload


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def orchestrator(fake_redis):
    return Orchestrator(redis_client=fake_redis)


@pytest.fixture
def sample_payload():
    return RegistrationPayload(
        url="http://model-service:8080",
        info={"name": "test-model", "version": "1.0.0"},
    )


class TestRegister:
    def test_register_returns_registration_response(self, orchestrator, sample_payload):
        response = orchestrator.register(sample_payload)

        assert response.status == "registered"
        assert response.service_url == sample_payload.url
        assert response.ttl_seconds == DEFAULT_TTL_SECONDS
        assert response.id  # ULID should be generated
        assert "$ping" in response.ping_url

    def test_register_stores_service_in_redis(self, orchestrator, sample_payload, fake_redis):
        response = orchestrator.register(sample_payload)

        key = f"service:{response.id}"
        assert fake_redis.exists(key)

    def test_multiple_registrations_create_unique_ids(self, orchestrator, sample_payload):
        response1 = orchestrator.register(sample_payload)
        response2 = orchestrator.register(sample_payload)

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

    def test_get_all_returns_registered_services(self, orchestrator, sample_payload):
        orchestrator.register(sample_payload)
        orchestrator.register(sample_payload)

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
