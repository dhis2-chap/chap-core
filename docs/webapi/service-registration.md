# Service Registration (v2 API)

!!! warning "Experimental Feature"
    This feature is a work in progress and considered experimental. The API may change in future releases without prior notice.

The v2 API provides service registration endpoints that enable ML models built with [chapkit](https://github.com/dhis2-chap/chapkit) and [servicekit](https://github.com/winterop-com/servicekit) to register themselves with CHAP for automatic discovery.

## Overview

Services register with the CHAP orchestrator and must send periodic keepalive pings to maintain their registration. Services that fail to ping within the TTL window are automatically expired.

Key features:

- **Idempotent registration** - Re-registering a service with the same ID returns existing data
- **Slug-based IDs** - Services provide their own unique identifier (lowercase, hyphens, numbers)
- **TTL-based expiry** - Services must ping periodically to stay registered (default: 30 seconds)
- **Redis/Valkey backend** - Service data stored with automatic expiration

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v2/services/$register` | Required | Register a service |
| PUT | `/v2/services/{id}/$ping` | Required | Keepalive ping |
| GET | `/v2/services` | Public | List all services |
| GET | `/v2/services/{id}` | Public | Get single service |
| DELETE | `/v2/services/{id}` | Required | Deregister service |

## Authentication

Protected endpoints require an API key in the `X-Service-Key` header.

### Server Configuration

Set the `SERVICEKIT_REGISTRATION_KEY` environment variable on the CHAP server:

```bash
export SERVICEKIT_REGISTRATION_KEY="your-secret-key"
```

### Client Usage

Include the key in requests to protected endpoints:

```python
import httpx

headers = {"X-Service-Key": "your-secret-key"}

response = httpx.post(
    "http://chap-server/v2/services/$register",
    json=payload,
    headers=headers,
)
```

### Error Responses

| Scenario | Response |
|----------|----------|
| Missing header | 422 Unprocessable Entity |
| Invalid key | 401 Unauthorized |
| Key not configured on server | 503 Service Unavailable |

## Registration Payload

The registration payload contains the service URL and MLServiceInfo metadata:

```python
{
    "url": "http://my-model:8080",
    "info": {
        "id": "my-model",  # Unique slug identifier
        "display_name": "My ML Model",
        "version": "1.0.0",
        "description": "A predictive model for disease forecasting",
        "model_metadata": {
            "author": "Your Name",
            "organization": "Your Org"
        },
        "period_type": "monthly",  # or "weekly"
        "min_prediction_periods": 1,
        "max_prediction_periods": 12
    }
}
```

## Registration Response

```python
{
    "id": "my-model",
    "status": "registered",
    "service_url": "http://my-model:8080",
    "message": "Service registered successfully",
    "ttl_seconds": 30,
    "ping_url": "/v2/services/my-model/$ping"
}
```

## Keepalive Mechanism

Services must send periodic pings to maintain their registration:

```python
response = httpx.put(
    "http://chap-server/v2/services/my-model/$ping",
    headers={"X-Service-Key": "your-secret-key"},
)
```

The ping resets the TTL timer. If a service fails to ping within the TTL window (default 30 seconds), it is automatically removed from the registry.

## Integration with Servicekit

[Servicekit](https://github.com/winterop-com/servicekit) handles registration automatically. Configure your service with the registration URL and key:

```python
from servicekit import Service, ServiceInfo

service = Service(
    info=ServiceInfo(
        id="my-model",
        display_name="My Model",
        description="Disease prediction model",
    ),
    chap_url="http://chap-server",
)

# Registration and keepalive are handled automatically
service.run()
```

Set the environment variable for authentication:

```bash
export SERVICEKIT_REGISTRATION_KEY="your-secret-key"
```

See the [servicekit documentation](https://github.com/winterop-com/servicekit) for more details.

## Integration with Chapkit

[Chapkit](https://github.com/dhis2-chap/chapkit) extends servicekit with ML-specific functionality. The `MLServiceInfo` schema includes additional fields for model metadata:

```python
from chapkit import MLService, MLServiceInfo, ModelMetadata

service = MLService(
    info=MLServiceInfo(
        id="dengue-predictor",
        display_name="Dengue Predictor",
        model_metadata=ModelMetadata(
            author="Research Team",
            organization="Health Institute",
        ),
        period_type="monthly",
    ),
    chap_url="http://chap-server",
)

service.run()
```

See the [chapkit documentation](https://github.com/dhis2-chap/chapkit) for building ML prediction services.
